//
//  NER_LSTM.hpp
//  NER
//
//  Created by 潘洪岩 on 2020/1/5.
//  Copyright © 2020 潘洪岩. All rights reserved.
//

#ifndef NER_LSTM_hpp
#define NER_LSTM_hpp

#include <stdio.h>
#include <iostream>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/platform/env.h>
#include <tensorflow/cc/saved_model/loader.h>
#include <tensorflow/cc/saved_model/signature_constants.h>
#include <tensorflow/core/kernels/lookup_table_init_op.h>
#include <tensorflow/core/kernels/lookup_table_op.h>

#include "encoder/Encoder.h"
#include "decoder/Decoder.h"

using namespace tensorflow;
using namespace encoder;

class NER_LSTM
{
private:
    std::string model;
    Tensor dropout;
    float dropout_value=1.0;
    string kSavedModelTagServe = "serve";
    
    encoder::SequenceEncoder encoder;
    decoder::Decoder decoder;
    
    std::unique_ptr<tensorflow::Session> sess;
    tensorflow::SessionOptions sess_options;
    tensorflow::RunOptions run_options;
    tensorflow::SavedModelBundle bundle;
    
    tensorflow::Status status;
public:
    NER_LSTM(std::string &model_path,SequenceEncoder &encoder):model(model_path),encoder(encoder){
        Tensor dropout(DT_FLOAT,TensorShape({1}));
        auto data=dropout.flat<float>().data();
        std::vector<float> d({dropout_value});
        copy_n(d.begin(),1,data);
        bool s = NER_LSTM::dropout.CopyFrom(dropout, TensorShape({1}));
        if(s)
            LOG(INFO) << "Dropout set:" <<dropout_value;
    }
    
    bool LoadModel()
    {
        status=tensorflow::LoadSavedModel(sess_options, run_options, model, { NER_LSTM::kSavedModelTagServe }, &bundle);
        LOG(INFO) << "Model: "<<model<<" Load Status "<<status.ToString();
        sess=std::move(bundle.session);
        return status.ok();
    }
    
    Tensor ConvOneTensor(std::vector<int> &s_codes);
    
    std::map<std::string,std::string> Predict(std::string &sentence);
    
    ~NER_LSTM()
    {
        status = sess->Close();
        LOG(INFO)<< "Close TFSession:"<<status.ToString();
    }
};


#endif /* NER_LSTM_hpp */
