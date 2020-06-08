//
//  NER_LSTM.cpp
//  NER
//
//  Created by 潘洪岩 on 2020/1/5.
//  Copyright © 2020 潘洪岩. All rights reserved.
//

#include "NER_LSTM.hpp"

Tensor NER_LSTM::ConvOneTensor(std::vector<int> &s_codes)
{
    long size=s_codes.size();
    Tensor x(DT_INT32,TensorShape({size}));
    auto v=x.flat<int>().data();
    copy_n(s_codes.begin(),size,v);
    return x;
}

std::map<std::string,std::string> NER_LSTM::Predict(std::string &sentence)
{
    std::map<std::string,std::string> entities;
    std::wstring stc=boost::locale::conv::utf_to_utf<wchar_t>(sentence);
    long length=stc.length();
//    std::cout<<length<<std::endl;
    Tensor x(DT_INT32,TensorShape({1,length}));
    encoder.Encode(stc, x,length);
    std::vector<int> sl({(int)length});
    Tensor l=ConvOneTensor(sl);
    std::vector<std::pair<std::string, Tensor>> inputs = {
        {"word_ids",x},
        {"sequence_lengths",l},
        {"dropout",dropout}
    };
    std::vector<Tensor> outputs;
    if(status.ok()){
        Status status = sess->Run(inputs, {"crf_decode/output_labels"}, {}, &outputs);
        if(status.ok()){
            Tensor result=outputs[0];
//            std::cout<<result.tensor<int, 2>()<<std::endl;
            entities =decoder.decode(stc, result, length);
        }else{
            LOG(INFO) << "NER Predict Failed:"<<status.error_message()<<std::endl;
        }
    }
    return entities;
}
