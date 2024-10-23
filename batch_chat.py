import copy

from typing import (
    List,
    Optional,
    Tuple,
    Union
)
import torch

from transformers import (
    PreTrainedTokenizer,
    AutoModelForCausalLM,
    GenerationConfig

)

from Qwen_1_8B_Chat.modeling_qwen import (
     _SENTINEL,
     _ERROR_STREAM_IN_CHAT,
     _ERROR_BAD_CHAT_FORMAT
)

from Qwen_1_8B_Chat.qwen_generation_utils import (
     HistoryType,
     make_context,
     decode_tokens,
     get_stop_words_ids,
     StopWordsLogitsProcessor
)

def batch_chat(
        model: AutoModelForCausalLM,
        tokenizer: PreTrainedTokenizer,
        query: Union[str, List[str]],
        history: Optional[HistoryType],
        system: str = "You are a helpful assistant.",
        stream: Optional[bool] = _SENTINEL,
        stop_words_ids: Optional[List[List[int]]] = None,
        generation_config: Optional[GenerationConfig] = None,
        **kwargs,
    ) -> Tuple[str, HistoryType]:
    """
    This is the batched version of original modeling_qwen.chat
    """
    generation_config = generation_config if generation_config is not None else model.generation_config

    assert stream is _SENTINEL, _ERROR_STREAM_IN_CHAT
    #assert generation_config.chat_format == 'chatml', _ERROR_BAD_CHAT_FORMAT
    generation_config.chat_format = 'raw'

    assert history is None, "We only support None history now"
    if history is None:
        history = []
    else:
        # make a copy of the user's input such that is is left untouched
        history = copy.deepcopy(history)

    if stop_words_ids is None:
        stop_words_ids = []

    max_window_size = kwargs.get('max_window_size', None)
    if max_window_size is None:
        max_window_size = generation_config.max_window_size
    
    if isinstance(query, str):
        batch_query = [query]
    elif isinstance(query, list):
        batch_query = copy.deepcopy(query)
    
    batch_raw_text, batch_context_tokens = [],[]
    # TODO: limit the maximum length of input text
    batch_input_ids = tokenizer(batch_query, padding=True, return_tensors='pt').to(model.device)
    batch_input_ids = batch_input_ids['input_ids']

    #for query in batch_query:
    #    # Let's use raw mode
    #    #raw_text, context_tokens = make_context(
    #    #    tokenizer,
    #    #    query,
    #    #    history=None,
    #    #    system=system,
    #    #    max_window_size=max_window_size,
    #    #    chat_format=generation_config.chat_format,
    #    #)
    #    batch_raw_text.append(raw_text)
    #    batch_context_tokens.append(context_tokens)

    stop_words_ids.extend(get_stop_words_ids(
        generation_config.chat_format, tokenizer
    ))

    #input_ids = torch.tensor([context_tokens]).to(self.device)
    outputs = model.generate(
                batch_input_ids,
                stop_words_ids=stop_words_ids,
                return_dict_in_generate=False,
                generation_config=generation_config,
                **kwargs,
            )
    
    # Check result and decode each generated results 
    
    response_list = []
    for query_idx in range(len(batch_query)):
        response = decode_tokens(
            outputs[query_idx],
            tokenizer,
            raw_text_len=len( batch_query[query_idx] ),
            context_length=len( batch_input_ids[query_idx] ),
            #chat_format=generation_config.chat_format,
            chat_format="raw",
            verbose=False,
            errors='replace'
        )
        response_list.append(response)

    # as history is a copy of the user inputs,
    # we can always return the new turn to the user.
    # separating input history and output history also enables the user
    # to implement more complex history management
    #history.append((query, response))

    return response_list, None