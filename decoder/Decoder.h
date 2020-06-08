//
//  Decoder.h
//  NER
//
//  Created by 潘洪岩 on 2020/1/5.
//  Copyright © 2020 潘洪岩. All rights reserved.
//

#ifndef Decoder_h
#define Decoder_h

#include <iostream>
#include <string>
#include <boost/locale.hpp>

namespace ner_label{
//const std::vector<std::string> TAG2LABEL={"0","B-TARGET","I-TARGET","B-BEDDING","I-BEDDING"};
//const int[] LABEL={0,1,2,3,4};
//{{"O", 0}, {"B-ADDRESS", 1}, {"I-ADDRESS", 2}, {"B-BOOK", 3}, {"I-BOOK", 4}, {"B-COMPANY", 5}, {"I-COMPANY", 6}, {"B-GAME", 7}, {"I-GAME", 8}, {"B-GOVERNMENT", 9}, {"I-GOVERNMENT", 10},{"B-MOVIE", 11}, {"I-MOVIE", 12}, {"B-NAME", 13}, {"I-NAME", 14}, {"B-ORGANIZATION", 15}, {"I-ORGANIZATION", 16}, {"B-POSITION", 17}, {"I-POSITION", 18}, {"B-SCENE", 19}, {"I-SCENE", 20}};

const static std::vector<std::pair<std::string,int>> TAG2LABEL = {{"O", 0}, {"ADDRESS", 1}, {"ADDRESS", 2}, {"BOOK", 3}, {"BOOK", 4}, {"COMPANY", 5}, {"COMPANY", 6}, {"GAME", 7}, {"GAME", 8}, {"GOVERNMENT", 9}, {"GOVERNMENT", 10},{"MOVIE", 11}, {"MOVIE", 12}, {"NAME", 13}, {"NAME", 14}, {"ORGANIZATION", 15}, {"ORGANIZATION", 16}, {"POSITION", 17}, {"POSITION", 18}, {"SCENE", 19}, {"SCENE", 20}};

const int TARGET_SIZE=2;
enum START_INDEX{TARGET=1,ASPECT=3};
struct SCAN_TAG{
    int target=1;
    int aspect=1;
};

}

namespace decoder {

class Decoder
{
    
public:
    std::map<std::string,std::string> decode(std::wstring &stc, Tensor &result,long length);
    
    std::vector<int> VerticalVector(int tag_start_index, Tensor &inputs,long size);
    
    void GetToken(std::wstring &stc,std::vector<int> &tags,int tag_type,std::map<std::string,std::string> &result);
};

}

inline std::map<std::string,std::string> decoder::Decoder::decode(std::wstring &stc, Tensor &result, long size){
    std::map<std::string,std::string> target_result;
    auto tags=result.flat<int>();
    //    int tag_category_num=oa_label::TARGET_SIZE;//扫描目标类别数
    //    oa_label::SCAN_TAG scan;//扫描目标类
    for(int i=0;i<size;i++){
        int tag = tags(i);
        std::wstring tag_word;
        if(tag==1||(tag%2==1))
        {
            tag_word.push_back(stc[i]);
            for(i+=1;i<size;i++)
            {
                if((tags(i)-tag)==1){
                    tag_word.push_back(stc[i]);
                }else{// if(tags(i)!=0)
                    i-=1;
                    break;
                }
            }
            if(tag_word.length()>1){
                std::string label=ner_label::TAG2LABEL[tag].first;
                std::string w_tag=boost::locale::conv::utf_to_utf<char>(tag_word);
                target_result.insert(std::make_pair(w_tag,label));
            }
            tag_word.clear();
        }
    }
    return target_result;
}

inline std::vector<int> decoder::Decoder::VerticalVector(int tag_start_index, Tensor &inputs, long size){
    std::vector<int> vertical_vector(size);
    auto tags=inputs.flat<int>();
    for(int i=0;i<size;i++){
        int t=tags(i)-tag_start_index;
        vertical_vector[i]=t;
    }
    return vertical_vector;
}

inline void decoder::Decoder::GetToken(std::wstring &stc, std::vector<int> &tags,int tag_type,std::map<std::string,std::string> &result){
    std::string tag=ner_label::TAG2LABEL[tag_type].first;
    std::wstring tag_word;
    
    for(int i=0;i<tags.size();i++)
    {
        if(tags[i]==0){
            tag_word.push_back(stc[i]);
            for(int j=i+1;j<tags.size();j++){
                if(tags[j]==1)
                    tag_word.push_back(stc[j]);
                else
                    break;
            }
            if(tag_word.length()>1){
                std::string w_tag=boost::locale::conv::utf_to_utf<char>(tag_word);
                result.insert(std::make_pair(w_tag,tag));
            }
            tag_word.clear();
        }
    }
}

#endif /* Decoder_h */
