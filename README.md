# TF-NER:  A C++ dylib for Tensorflow: Doing NER
***This xcode project will generate a dynamic library. But also support other OS.***
***由Python生成.pb模型后，采用C++封装的方式，完成方便的部署方式，这里封装成动态库，方便各语言对模型的调用。***
> ###导出函数:
>（1） NER_INIT为加载模型参数，指定Python生成的saved_mode.pb目录，中同时包含words.txt的词表，为输入内容提供查表操作。
>（2）NER_GET出入字符串，得到格式化的结果串
> ```c++
> /** @data: model path; 指定包含words.txt,saved_model.pb和variables的目录
>  ** @vocab_size: vocab size; word2.txt 大小
>  ** @UNK_id: unkown word id in vocab; words.txt中unk的id
>  ** @return: loaded model state
>  **/ 
> bool NER_INIT(const char *data,int vocab_size,int UNK_id)
> /** @content: input
>  ** @return: ner result string
>  **/
> const char* NER_GET(const char* content)
> ```

## 1.Encoder
**中文在C/C++环境中，不同于英文字符，这里将使用wchar_t，宽字符来处理中文字符序列。然后将字符转化为机器码做HashTable，从而将模型输入和输出全部映射为数字。**

例：TensorFlow中实现查表操作得到tensor

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

## 2.C++ Session Run "serve"

### 加载模型

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

### 预测

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

### Label信息还原
```c++
//内联函数还原label id对应标签，同样字符以wchar_t处理。
inline std::map<std::string,std::string> decoder::Decoder::decode(std::wstring &stc, Tensor &result, long size)
```

## Tensorflow 1.13.2: C++ Include & Lib

[Mac](链接: https://pan.baidu.com/s/1bctxhoGw3Y2AMH0Af9rzPQ  密码: 011j)

***不同平台的硬件可能需要重新编译TensorFlow的c++动态库，以支持相应硬件，编译方式参考[TensorFlow官方文档](https://tensorflow.google.cn/install)，当前方式推荐使用bazel编译***

备：需要C++ Boost库依赖
