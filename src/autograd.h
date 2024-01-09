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

    virtual void eval();

    ts::Tensor &getVal(){
        if(value == nullptr) this->eval();
        return *value;
    };

    shared_ptr<ts::Tensor> value_ptr(){
        return value;
    }

    shared_ptr<Node> grad_ptr(){
        return grad;
    }

    vector<shared_ptr<Node>> parents_ptr(){
        return parents;
    }

    vector<shared_ptr<Node>> children_ptr(){
        return children;
    }

    friend ostream &operator<<(ostream &os, Node &node);
    friend Node operator+(Node &node1, Node &node2);
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
    Add_node(Node &a, Node &b) throw();
    Add_node(Node a){
        value = a.value_ptr();
        grad = a.grad_ptr();
        parents = a.parents_ptr();
        children = a.children_ptr();
    }

    void outterEval();

    virtual void eval() throw();
};


};

