

#include <cassert>
#include <iostream>
#include <unordered_map>
#include <string>


// basic file operations
#include <iostream>
#include <fstream>

#include <stdio.h>
#include <stdlib.h>
using namespace std;

int main () {

    ofstream myfile;
    // char* str = itoa(13);
    // string a = "122";
    
    string s =  to_string(122222) + ".txt";

    myfile.open (s);
    myfile << "Writing this to a file.\n";
    myfile.close();
    return 0;

}