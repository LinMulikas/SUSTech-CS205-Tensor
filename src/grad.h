#pragma once
#include <memory>
#include <iostream>

namespace grad{
class Node;
};

namespace ts{
class Tensor;

Tensor add(Tensor &ts1, Tensor &ts2) throw();


};

namespace grad{
using std::make_shared, std::shared_ptr, std::vector;
using std::initializer_list;
using std::ostream, std::cin, std::cout, std::endl;

class Node{
protected:
    shared_ptr<ts::Tensor> value;
    shared_ptr<Node> x;
    shared_ptr<Node> grad;
    vector<shared_ptr<Node>> parents;
    vector<shared_ptr<Node>> children;

public:
    Node(ts::Tensor &ts);

    Node(vector<shared_ptr<Node>> &pars);

    Node(const Node &that);

    Node(){
        value = nullptr;
        x = nullptr;
        grad = nullptr;
        parents = {};
        children = {};
    };

    ts::Tensor &getTensor(){
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

    virtual void eval();

    virtual grad::Node gradTo(Node &that) throw();
};

class Variable : public Node{
public:
    Variable(ts::Tensor &ts);

    Variable(){};

    virtual void eval(){};

    virtual grad::Node gradTo(Node &that) throw();

};

/*
    The add-node has parents with size 2.
*/
class AddNode : public Node{
public:
    AddNode(shared_ptr<Node> node1, shared_ptr<Node> node2);

    AddNode(Node &a, Node &b);

    AddNode(Node a){
        value = a.value_ptr();
        grad = a.grad_ptr();
        parents = a.parents_ptr();
        children = a.children_ptr();
    }

    virtual void eval() throw();

    virtual grad::Node gradTo(Node &that) throw();

};

class SubNode : public Node{
public:
    SubNode(shared_ptr<Node> node1, shared_ptr<Node> node2);

    SubNode(Node &a, Node &b);

    SubNode(Node a){
        value = a.value_ptr();
        grad = a.grad_ptr();
        parents = a.parents_ptr();
        children = a.children_ptr();
    }

    virtual void eval() throw();

    virtual grad::Node gradTo(Node &that) throw();
};

class SinNode : public Node{
public:
    SinNode(Node &node);

    SinNode(ts::Tensor &ts);

    SinNode(){};

    virtual void eval() throw();

    virtual grad::Node gradTo(Node &that) throw();

};

grad::Node autograd(ts::Tensor &in, ts::Tensor &out) throw();

};
