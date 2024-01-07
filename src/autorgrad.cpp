#include <vector>
#include <memory>
#include <iostream>
#include "tensor.h"

using std::cout, std::endl, std::vector, std::shared_ptr, std::ostream, std::make_shared;

class Node{
protected:
    shared_ptr<double> value;
    shared_ptr<Node> grad;
    vector<shared_ptr<Node>> parents;
    vector<shared_ptr<Node>> children;

public:
    Node(vector<shared_ptr<Node>> &pars){
        parents = pars;
    }

    Node(){
        value = nullptr;
        grad = nullptr;
        parents = {};
        children = {};
    }

    void add_parent(Node par_node){
        parents.push_back(make_shared<decltype(par_node)>(par_node));
    }

    virtual void eval(){
        cout << "No eval() override." << endl;
    };

    double getVal(){
        if(value == nullptr) this->eval();
        return *value;
    }

    // Node(const Node &that){
    //     value = that.value;
    //     grad = that.grad;
    //     parents = that.parents;
    //     children = that.children;
    // }


    friend ostream &operator<<(ostream &os, Node &var){
        if(var.value == nullptr) var.eval();
        os << var.getVal();
        return os;
    }

};

class Variable : public Node{
private:
    double val;

public:
    Variable(double x) : val(x){}

    virtual void eval(){
        if(value == nullptr) value.reset(&val);
    }

};

class Add : public Node{
public:
    Add(Variable a, Variable b){
        parents.push_back(make_shared<Variable>(a));
        parents.push_back(make_shared<Variable>(b));
    }

    virtual void eval(){
        if(value == nullptr){
            value = make_shared<double>((double)0);
            for(auto par_node : parents){
                par_node->eval();
                *value += par_node->getVal();
            }
        }
    }


};