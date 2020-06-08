//
//  Encoder.h
//  NER
//
//  Created by 潘洪岩 on 2020/1/5.
//  Copyright © 2020 潘洪岩. All rights reserved.
//

#ifndef Encoder_h
#define Encoder_h

#include <stdio.h>
#include <iostream>
#include <vector>
#include <tensorflow/core/kernels/lookup_table_init_op.h>
#include <tensorflow/core/kernels/lookup_table_op.h>
#include <boost/locale.hpp>

using namespace tensorflow;

namespace encoder {
enum LABLE{ UNK = -4,ENG = -3,NUM = -2,PAD = -1 };

class SequenceEncoder {
    
private:
    Tensor default_v;
    std::string vocab;
    tensorflow::Env* env=tensorflow::Env::Default();
    tensorflow::lookup::HashTable<int32, int32> *table;
public:
    SequenceEncoder(std::string &vocab_file,int default_value):vocab(vocab_file){
        Tensor dflt(DT_INT32,TensorShape({1}));
        std::vector<int> data({default_value});
        auto d=dflt.flat<int>().data();
        std::copy_n(data.begin(),1,d);
        bool s = default_v.CopyFrom(dflt, TensorShape({1}));
        if(s)
            LOG(INFO)<<"Set <UNK> Default_value:"<<default_value;
        table=new tensorflow::lookup::HashTable<int32, int32>(nullptr,nullptr);
    }
    
    void LoadVocab(int vocab_size,char delimiter,int key_index,int value_index){
        tensorflow::Status status=tensorflow::lookup::InitializeTableFromTextFile(vocab, vocab_size, delimiter, key_index, value_index, env, table);
        LOG(INFO) <<"Load Vocab HashTable: "<<status.ToString()<<" ;";
    }
    
    size_t Encode(std::wstring &stc,Tensor &values,long length)
    {
        std::vector<int> s_codes(length);
        for(int i=0;i<length;i++)
        {
            if(stc[i]>=48 && stc[i]<=57)
                s_codes[i]=NUM;
            else if((stc[i]>=65 && stc[i]<=90)||((stc[i]>=97 && stc[i]<=122)))
                s_codes[i]=ENG;
            else
                s_codes[i]=stc[i];
        }
        Tensor keys(DT_INT32,TensorShape({1,length}));
        auto k_data=keys.flat<int>().data();
        std::copy_n(s_codes.begin(),length,k_data);
        
        tensorflow::Status status = table->Find(nullptr, keys, &values, default_v);
        if(!status.ok())
            LOG(ERROR)<<"HashTable:"<<status.error_message();
        return s_codes.size();
    }
    
    ~SequenceEncoder(){
        //table->~HashTable();
    }
};
}

#endif /* Encoder_h */
