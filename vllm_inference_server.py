from vllm_wrapper import vLLMWrapper

if __name__ == '__main__':
    #model = "Qwen/Qwen-72B-Chat-Int4"
    model = "/data/hg_models/Qwen-72B-Chat-Int4"

    vllm_model = vLLMWrapper(model,
                             quantization = 'gptq',
                             dtype="float16",
                             tensor_parallel_size=4)

    response, history = vllm_model.chat(query="你好",
                                        history=None)
    print(response)
    response, history = vllm_model.chat(query="给我讲一个年轻人奋斗创业最终取得成功的故事。",
                                        history=history)
    print(response)
    response, history = vllm_model.chat(query="给这个故事起一个标题",
                                        history=history)
    print(response)
