#include <variant>
#include <iostream>
#include<string>
using namespace std;

class vec{
public:
  int x;
  int y;

public:
 vec(){}
 vec(int a,int b):x(a),y(b){}

void print(){
    std::cout<<x<<","<<y<<std::endl;
}

};

int main(){
   
   char* s=new char[2];
   s[0]='a';
   s[1]='b';
 
    
    
    return 0;
}