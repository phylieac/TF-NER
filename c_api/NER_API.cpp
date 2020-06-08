//
//  NER_API.cpp
//  NER
//
//  Created by 潘洪岩 on 2020/1/11.
//  Copyright © 2020 潘洪岩. All rights reserved.
//

#include "NER_API.hpp"
#include "tf/model/ner/NER_LSTM.hpp"

NER_LSTM *ner;

bool NER_INIT(const char *data,int vocab_size,int UNK_id)
{
    std::string model_path(data);
    std::string vocab_file=model_path+"/words.txt";
    encoder::SequenceEncoder encoder(vocab_file,UNK_id);
    encoder.LoadVocab(vocab_size, '\t', 0, 1);
    bool s=false;
    ner = new NER_LSTM(model_path,encoder);
    s=ner->LoadModel();
    return s;
}

const char* NER_GET(const char* content)
{
    std::string sentence(content);
    std::map<std::string,std::string> rslt;
    rslt=ner->Predict(sentence);
    std::string result;
    for(auto entity:rslt)
    {
        result+=entity.first+"/"+entity.second+"#";
    }
    long length=result.length()+1;
    char *data=new char[length];
    std::strcpy(data, result.c_str());
    return data;
}
