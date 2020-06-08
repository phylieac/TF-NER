# TF-NER:  A C++ dylib for Tensorflow: Doing NER
***This project will generate a dynamic library.***
> ###Export Funtion:
> ```c++
> /** @data: model path
>  ** @vocab_size: vocab size
>  ** @UNK_id: unkown word id in vocab
>  ** @return: loaded model state
>  **/ 
> bool NER_INIT(const char *data,int vocab_size,int UNK_id)
> /** @content: input
>  ** @return: ner result string
>  **/
> const char* NER_GET(const char* content)
> ```

# 1.Encoder part
**中文在C/C++环境中，不同于英文字符，这里将使用wchar_t，宽字符来处理中文字符序列。然后将字符转化为机器码做HashTable，从而将模型输入和输出全部映射为数字。**

例：TensorFlow中实现查表操作得到Tensor

TensorFlow HashTable：

```c++

#include <tensorflow/core/kernels/lookup_table_init_op.h>
#include <tensorflow/core/kernels/lookup_table_op.h>

// 获取HashTable对象
tensorflow::lookup::HashTable<int32, int32> *table = new tensorflow::lookup::HashTable<int32, int32>(nullptr,nullptr);

// 初始化HashTable
void LoadVocab(int vocab_size,char delimiter,int key_index,int value_index){
    tensorflow::Status status=tensorflow::lookup::InitializeTableFromTextFile(vocab, vocab_size, delimiter, key_index, value_index, env, table);
    LOG(INFO) <<"Load Vocab HashTable: "<<status.ToString()<<" ;";
}

// Lookup查表
tensorflow::Status status = table->Find(nullptr, keys, &values, default_v);
if(!status.ok())
    LOG(ERROR)<<"HashTable:"<<status.error_message();
```

# 2.C++ Session Run "serve"

## 加载模型

```c++
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/platform/env.h>
#include <tensorflow/cc/saved_model/loader.h>
#include <tensorflow/cc/saved_model/signature_constants.h>


string kSavedModelTagServe = "serve";

std::unique_ptr<tensorflow::Session> sess;
tensorflow::SessionOptions sess_options;
tensorflow::RunOptions run_options;
tensorflow::SavedModelBundle bundle;

//加载模型
status=tensorflow::LoadSavedModel(sess_options, run_options, model, { NER_LSTM::kSavedModelTagServe }, &bundle);
sess=std::move(bundle.session);
```

## 预测

```c++
//PlaceHolder
std::vector<std::pair<std::string, Tensor>> inputs = {
    {"word_ids",x},
    {"sequence_lengths",l},
    {"dropout",dropout}
};

//Outputs
std::vector<Tensor> outputs;
Status status = sess->Run(inputs, {"crf_decode/output_labels"}, {}, &outputs);
Tensor result=outputs[0];
std::cout<<result.tensor<int, 2>()<<std::endl;//Result Label ID
```

## Label信息还原
```c++
//内联函数还原label id对应标签，同样字符以wchar_t处理。
inline std::map<std::string,std::string> decoder::Decoder::decode(std::wstring &stc, Tensor &result, long size)
```
