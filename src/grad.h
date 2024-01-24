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
public:
    shared_ptr<ts::Tensor> value;
    shared_ptr<ts::Tensor> grad;

    Node() : value(nullptr), grad(nullptr) {}
    Node(ts::Tensor &v, ts::Tensor &g);
    virtual ~Node() = default;

    virtual void forward() = 0;

    virtual void backward() = 0;
};

class ValueNode : public Node{
public:
    ValueNode();
    ValueNode(ts::Tensor & ts);
    void forward() override {}
    void backward() override {}
};

class AddNode : public Node{
private:
    shared_ptr<Node> lhs, rhs;
public:
    AddNode(shared_ptr<Node> l, shared_ptr<Node> r) : lhs(l), rhs(r) {}

    void forward() override;

    void backward() override;
};

class SubNode : public Node{
private:
    shared_ptr<Node> lhs, rhs;
public:
    SubNode(shared_ptr<Node> l, shared_ptr<Node> r) : lhs(l), rhs(r) {}

    void forward() override;

    void backward() override;
};

class MulNode : public Node{
private:
    shared_ptr<Node> lhs, rhs;
public:
    MulNode(shared_ptr<Node> l, shared_ptr<Node> r) : lhs(l), rhs(r) {}

    void forward() override;

    void backward() override;
};

class DivNode : public Node{
private:
    shared_ptr<Node> lhs, rhs;
public:
    DivNode(shared_ptr<Node> l, shared_ptr<Node> r) : lhs(l), rhs(r) {}

    void forward() override;

    void backward() override;
};

};
