#include <iostream>
#include <vector>
#include "tensor.h"
#include "autograd.h"

namespace ts{
using std::vector;

// VariantData
using VariantData = std::variant<bool, int, float, double>;

VariantData Tensor::copy_tile(VariantData *src, Tensor *dst, int idx, int *src_shape, int dim){
    int subarrays = 1; // 当前维度的子数组数量
    for(int d = dim + 1; d < dst->dimension; ++d){
        subarrays *= dst->shape.get()[d];
    }
    int cur_id = (idx / subarrays) % src_shape[dim];
    if(dim < dst->dimension - 1)
        return copy_tile(src + cur_id * src_shape[dim], dst, idx % subarrays, src_shape, dim + 1);
    else
        return src[idx % src_shape[dim]];
}

/*
    Get the val with most promotion(double).
*/
double promote(VariantData &vd){
    // std::cout << "promote" << std::endl;
    switch(vd.index()){
    case 0:
        return (double)(*(std::get_if<0>(&vd)));
    case 1:
        return (double)(*(std::get_if<1>(&vd)));
    case 2:
        return (double)(*(std::get_if<2>(&vd)));
    case 3:
        return (double)(*(std::get_if<3>(&vd)));
    }
    return 0;
};


void assign(VariantData &vd, double val, int type_id){
    switch(type_id){
    case 0:
        vd = (bool)val;
    case 1:
        vd = (int)val;
    case 2:
        vd = (float)val;
    case 3:
        vd = (double)val;
    }
}


void Tensor::print(std::ostream &os, int index, int dim) const{
    if(dim == dimension){
        // 打印单个元素
        std::visit([&os](auto &&arg){
            os << std::fixed << std::setprecision(4) << arg;
                   }, data[index]);
    }
    else{
        // 打印开括号
        os << "[";

        int subarrays = 1; // 当前维度的子数组数量
        for(int d = dim + 1; d < dimension; ++d){
            subarrays *= shape.get()[d];
        }
        for(int i = 0; i < shape.get()[dim]; ++i){
            // 递归打印下一个维度
            print(os, index + i * subarrays, dim + 1);
            if(i < shape.get()[dim] - 1){
                os << ", ";
                if(dim < dimension - 1){  // 如果不是最内层维度，在逗号后换行
                    os << "\n" << std::string(dim + 1, ' ');  // 缩进以对齐
                }
            }
        }
        // 打印闭括号
        os << "]";
    }
}


int *Tensor::size(){
    return shape.get();
}

std::string Tensor::type_name(){
    switch(dtype_id){
    case 0:
        return "bool";
    case 1:
        return "int";
    case 2:
        return "float";
    case 3:
        return "double";
    }
}

VariantData *Tensor::data_ptr(){
    return data;
}

int Tensor::get_total_size(){
    return total_size;
}

Tensor::Tensor(int type_id){
    dimension = 0;
    shape = nullptr;
    data = nullptr;
    total_size = 0;
    dtype_id = type_id;
}


Tensor::Tensor(){
    dimension = 0;
    shape = nullptr;
    data = nullptr;
    total_size = 0;
    dtype_id = 1;
}

std::ostream &operator<<(std::ostream &os, const Tensor &t){
    if(t.dimension == 0 || t.total_size == 0){
        os << "[]";
        return os;
    }

    t.print(os, 0, 0);

    return os;
}


int Tensor::get_dimension() const{
    return dimension;
}

int *Tensor::get_shape() const{
    return shape.get();
}

Tensor cat(std::pair<Tensor, Tensor> &tensors, int dim){
    const int dimension = tensors.first.get_dimension();
    int size[dimension];
    for(int i = 0; i < dimension; i++){
        size[i] = tensors.first.get_shape()[i];
    }
    size[dim] += tensors.second.get_shape()[dim];
    Tensor t = Tensor::init_with_shape(size, dimension);
    for(int i = 0; i < t.get_total_size(); i++){
        if(i < tensors.first.get_total_size())
            t.data_ptr()[i] = tensors.first.data_ptr()[i];
        else{
            t.data_ptr()[i] = tensors.second.data_ptr()[i - tensors.first.get_total_size()];
        }
    }
    return t;
}

// The index operator can only get the index of the first dim.
Tensor Tensor::operator()(int idx){
    Tensor t = Tensor();
    if(dimension == 1){
        t.dimension = 1;
    }
    else t.dimension = dimension - 1;
    t.shape.reset(new int[t.dimension]);

    for(int i = 0; i < t.dimension; i++){
        if(dimension == 1){
            t.shape[i] = 1;
        }
        else{
            t.shape[i] = shape[i + 1];
        }
    }
    t.total_size = total_size / shape[0];

    t.data = t.total_size * idx + data;
    return t;
}

Tensor Tensor::operator()(int idx, std::pair<int, int> range){
    Tensor t = Tensor();
    t.dimension = dimension - 1;
    t.shape.reset(new int[t.dimension]);
    t.shape[0] = range.second - range.first;
    t.total_size = t.shape[0];
    for(int i = 1; i < t.dimension; i++){
        t.shape[i] = shape[i + 1];
        t.total_size *= shape[i + 1];
    }
    ;
    t.data = shape[0] * idx + data + range.first;
    return t;
}

void Tensor::operator=(const VariantData &value){
    for(int i = 0; i < total_size; i++){
        data[i] = value;
    }
}

int Tensor::cal_stride(int dim, int *shape){
    int stride = 1;
    for(int i = dim + 1; i < dimension; i++){
        stride *= shape[i];
    }
    return stride;
}


Tensor Tensor::transpose(int dim1, int dim2){
    // 检查维度是否有效
    if(dim1 < 0 || dim1 >= dimension || dim2 < 0 || dim2 >= dimension || dim1 == dim2){
        throw std::out_of_range("Invalid dimensions for transpose");
    }

    int old_dim1 = cal_stride(dim1, shape.get());
    int mod_dim1 = old_dim1 * shape[dim1];

    int old_dim2 = cal_stride(dim2, shape.get());
    int mod_dim2 = old_dim2 * shape[dim2];

    // 交换shape中的维度
    int *newShape = new int[dimension];
    for(int i = 0; i < dimension; ++i){
        newShape[i] = shape[i];
    }
    newShape[dim1] = shape[dim2];
    newShape[dim2] = shape[dim1];

    auto newData = new VariantData[total_size];

    int sub_dim1 = cal_stride(dim1, newShape);

    int sub_dim2 = cal_stride(dim2, newShape);
    // 把原来的坐标转化为新坐标，data[d1][...][d2] - data[d2][...][d1]差值计算
    for(int i = 0; i < total_size; ++i){
        int d1 = (i % mod_dim1) / old_dim1;
        int d2 = (i % mod_dim2) / old_dim2;
        int idx = i + d1 * (sub_dim2 - old_dim1) + d2 * (sub_dim1 - old_dim2);
        newData[idx] = data[i];
    }

    Tensor tensor = Tensor::init_with_shape(newShape, dimension);
    for(int i = 0; i < total_size; ++i){
        tensor.data_ptr()[i] = newData[i];
    }

    return tensor;
}

int calculateIndex(int *indices, int *strides, int dimension){
    int index = 0;
    for(int i = 0; i < dimension; ++i){
        index += strides[i] * indices[i];
    }
    return index;
}

int *getStrides(int *shape, int dimension){
    int *strides = new int[dimension];
    for(int i = 0; i < dimension; ++i){
        strides[i] = 1;
        for(int j = i + 1; j < dimension; ++j){
            strides[i] *= shape[j];
        }
    }
    return strides;
}

Tensor Tensor::permute(int dim[]){
    int *strides = getStrides(shape.get(), dimension);
    std::shared_ptr<int> newShape(new int[dimension]);

    for(int i = 0; i < dimension; ++i){
        newShape.get()[i] = shape[dim[i]];
    }
    int *newStrides = getStrides(newShape.get(), dimension);
    auto newArr = new VariantData[total_size];

    for(int i = 0; i < total_size; ++i){
        int *indices = new int[dimension];
        int idx = i;
        for(int j = 0; j < dimension; ++j){
            indices[j] = idx / strides[j];
            idx = idx % strides[j];
        }
        int *permuted_indices = new int[dimension];
        for(int j = 0; j < dimension; ++j){
            permuted_indices[j] = indices[dim[j]];
        }

        int newIdx = calculateIndex(permuted_indices, newStrides, dimension);
        newArr[newIdx] = data[i];
        delete[] indices;
        delete[] permuted_indices;
    }

    Tensor t = Tensor::init_with_shape(newShape.get(), dimension);
    for(int i = 0; i < total_size; ++i){
        t.data_ptr()[i] = newArr[i];
    }

    delete[] strides;
    delete[] newStrides;
    return t;
}

Tensor transpose(Tensor tensor, int dim1, int dim2){
    return tensor.transpose(dim1, dim2);
}

Tensor permute(Tensor tensor, int dim[]){
    return tensor.permute(dim);
}

Tensor sum(Tensor &ts, int dim);

Tensor operator+(Tensor &ts1, Tensor &ts2){
    return add(ts1, ts2);
}

Tensor add(Tensor &t1, Tensor &t2) throw(){
    int *shape1 = t1.get_shape();
    int *shape2 = t2.get_shape();
    size_t dim1 = sizeof(shape1) / sizeof(shape1[0]);
    size_t dim2 = sizeof(shape2) / sizeof(shape2[0]);

    // std::cout << dim1 << " " << dim2 << std::endl;
    if(dim1 != dim2){
        throw std::invalid_argument("The input tensors need to be same shape.");
        for(int i = 0; i < dim1; i++){
            if(shape1[i] != shape2[i]){
                throw std::invalid_argument("The input tensors need to be same shape.");
            }
        }
    }
    else{
        // Type promote
        Tensor result;

        int tid_1 = t1.get_dtype_id();
        int tid_2 = t2.get_dtype_id();
        int tid_rst = std::max(tid_1, tid_2);

        if(tid_rst == tid_1)result = zeros_like(t1);
        else result = zeros_like(t2);

        for(size_t i = 0; i < result.get_total_size(); i++){
            double v1 = promote(t1.data_ptr()[i]);
            double v2 = promote(t2.data_ptr()[i]);
            assign(result.data_ptr()[i], v1 + v2, tid_rst);
        }

        return result;
    }
}


}