#pragma once
#include <memory>
#include <iostream>

namespace ts{
class Tensor;
static Tensor add(Tensor &ts1, Tensor &ts2) throw();
};

namespace grad{
using std::make_shared, std::shared_ptr, std::vector;
using std::initializer_list;
using std::ostream, std::cin, std::cout, std::endl;


class Node{
protected:
    shared_ptr<ts::Tensor> value;
    shared_ptr<Node> grad;
    vector<shared_ptr<Node>> parents;
    vector<shared_ptr<Node>> children;

public:
    Node(vector<shared_ptr<Node>> &pars){
        parents = pars;
    };

    Node(const Node &that){
        value = that.value;
        grad = that.grad;
        parents = that.parents;
        children = that.children;
    };

    Node(){
        value = nullptr;
        grad = nullptr;
        parents = {};
        children = {};
    };

    void add_parent(Node par_node){
        parents.push_back(make_shared<decltype(par_node)>(par_node));
    };

    virtual void eval(){
        cout << "No eval() override." << endl;
    };

    ts::Tensor &getVal(){
        if(value == nullptr) this->eval();
        return *value;
    };


    friend ostream &operator<<(ostream &os, Node &node);
};

class Variable : public Node{
public:
    Variable(ts::Tensor &ts){
        value = make_shared<ts::Tensor>(ts);
    };

    Variable(){};

    virtual void eval(){};

};

/*
    The add-node has parents with size 2.
*/
class Add_node : public Node{
public:
    Add_node(Variable a, Variable b) throw();

    Add_node(shared_ptr<Node> a, shared_ptr<Node> b);
    Add_node(initializer_list<shared_ptr<Node>> nodes);

    Add_node(initializer_list<ts::Tensor> tensors);

    virtual void eval();
};


};

