

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <string>



#include <cstdio>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <array>
using namespace std;


std::string exec(const char* cmd) {
    std::array<char, 128> buffer;
    std::string result;
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd, "r"), pclose);
    if (!pipe) {
        throw std::runtime_error("popen() failed!");
    }
    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
        result += buffer.data();
    }
    return result;
}






int main(int argc, char** argv){
    // string s;
    // s = exec("sh run.sh");
    // cout<<s<<endl;

    FILE* ptr;
    char ch;

    char *buf = new char[64];
    int ofst;

    ptr = fopen(argv[1], "r");
    if (NULL == ptr) {
        fprintf(stderr, "file can't be opened \n");
        exit(-1);
    }            


    ch = fgetc(ptr);

    while (ch != '='){ch = fgetc(ptr);}

    ofst=0;
    while (ch != '\n'){
        ch = fgetc(ptr);
        buf[ofst++] = ch;
    }

    int N = atoi(buf);

    printf("N: %d\n", N);



    fclose(ptr);



}