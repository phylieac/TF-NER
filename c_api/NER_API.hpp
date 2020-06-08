//
//  NER_API.hpp
//  NER
//
//  Created by 潘洪岩 on 2020/1/11.
//  Copyright © 2020 潘洪岩. All rights reserved.
//

#ifndef NER_API_hpp
#define NER_API_hpp

#include <stdio.h>

#ifdef __WIN32__
#define NER_API extern "C" __declspec(dllexport)
#else
#define NER_API extern "C" __attribute__((visibility("default")))
#endif

NER_API bool NER_INIT(const char *data,int vocab_size=6753,int UNK_id=6752);
NER_API const char* NER_GET(const char *content);

#endif /* NER_API_hpp */
