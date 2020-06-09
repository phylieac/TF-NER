//
//  main.cpp
//  test
//
//  Created by 潘洪岩 on 2020/6/8.
//  Copyright © 2020 潘洪岩. All rights reserved.
//

#include <iostream>
#include <fstream>
#include "c_api/NER_API.hpp"

int main(int argc, const char * argv[]) {
    const char *data="/Users/panhongyan/git/zh-NER-TF/data_path_save/1591709661/checkpoints/model";
    bool s=NER_INIT(data,3613,3612);
    std::cout<<"初始化："<<s<<std::endl;
    
    std::string filename="/Users/panhongyan/test.data";
    std::ifstream file(filename);
    if(file.is_open())
    {
        std::string line;
        while (std::getline(file,line)) {
            std::cout<<line<<std::endl;
            std::clock_t start=std::clock();
            const char* result=NER_GET(line.c_str());
            std::clock_t stop=std::clock();
            std::cout<<(stop-start)/1000<<std::endl;
            std::cout<<result<<std::endl;
        }
    }
    file.close();
    
    return 0;
}
