#pragma once
#include <variant>
#include <iostream>
#include <type_traits>
#include <iomanip>
#include <vector>
#include <memory>
#include <random>
#include <typeinfo>
#include <initializer_list>
#include "autograd.h"

namespace ts{
using VariantData = std::variant<bool, int, float, double>;




using std::vector;
using std::shared_ptr, std::make_shared;

template<class T>
static int dtype_id_from(){
    if(typeid(T) == typeid(bool)) return 0;
    if(typeid(T) == typeid(int)) return 1;
    if(typeid(T) == typeid(float)) return 2;
    if(typeid(T) == typeid(double)) return 3;
    return -1;
}

static bool global_require_grad = false;

class Tensor{
private:
    std::shared_ptr<VariantData[]> data_shared;
    VariantData *data;

    int dimension;
    std::shared_ptr<int[]> shape;

    int total_size;
    int dtype_id;

    // Autograd
    bool _require_grad;
    shared_ptr<grad::Node> _node;

    template<typename ArrayType, size_t N>
    void copyData(ArrayType(&arr)[N], VariantData *&dest, VariantData *destEnd){
        for(size_t i = 0; i < N; ++i){
            if constexpr(std::is_array<ArrayType>::value){
                copyData(arr[i], dest, destEnd);
            }
            else{
                *dest = arr[i];
                if(dest < destEnd)
                    dest++;
            }
        }
    }

    template<typename ArrayType, size_t N>
    void getShape(ArrayType(&arr)[N], int dim){
        shape.get()[dim] = N;
        if constexpr(std::is_array<ArrayType>::value)
            getShape(arr[0], dim + 1);
    }


    void print(std::ostream &os, int index, int dim) const;

public:
    // public autograd
    void set_require_grad(bool require);
    void set_node(grad::Node node);
    void set_node(shared_ptr<grad::Node>);
    void init_node();

    friend Tensor operator+(Tensor &ts1, Tensor &ts2);

    static VariantData copy_tile(VariantData *src, Tensor *dst, int idx, int *src_shape, int dim);

    // init a tensor with exact const shape arr.
    template<size_t N>
    static Tensor init_with_shape(int(&size)[N]){
        Tensor t = Tensor();
        t.dimension = N;
        t.shape.reset(new int[t.dimension]);
        t.total_size = 1;
        for(int i = 0; i < t.dimension; i++){
            t.shape[i] = size[i];
            t.total_size *= size[i];
        }
        t.data_shared.reset(new VariantData[t.total_size]);
        t.data = t.data_shared.get();
        // default type float.
        t.dtype_id = 2;
        return t;
    }

    // init a tensor with exact shape arr and dim N.
    static Tensor init_with_shape(const int size[], int N){
        Tensor t = Tensor();
        t.dimension = N;
        t.shape.reset(new int[t.dimension]);
        t.total_size = 1;
        for(int i = 0; i < t.dimension; i++){
            t.shape[i] = size[i];
            t.total_size *= size[i];
        }
        t.data_shared.reset(new VariantData[t.total_size]);
        t.data = t.data_shared.get();

        // default type float.
        t.dtype_id = 2;

        return t;
    }

    // init a dtype T tensor with exact shape arr and dim N.
    // Have impled the autograd.
    template<typename T>
    static Tensor init_with_shape(const int size[], int N){
        Tensor t = Tensor(dtype_id_from<T>());

        t.dimension = N;
        t.shape.reset(new int[t.dimension]);
        t.total_size = 1;
        for(int i = 0; i < t.dimension; i++){
            t.shape[i] = size[i];
            t.total_size *= size[i];
        }
        t.data_shared.reset(new VariantData[t.total_size]);
        t.data = t.data_shared.get();


        t.dtype_id = dtype_id_from<T>();

        return t;
    }

    int *size();

    std::string type_name();

    int get_dtype_id(){
        return dtype_id;
    }

    VariantData *data_ptr();

    int get_total_size();

    int *get_shape() const;

    int get_dimension() const;

    void showShape(){
        for(int i = 0; i < dimension; i++){
            std::cout << shape[i] << " ";
        }
        std::cout << std::endl;
    }


    // Constructors
    // - Init tensor with exact data.
    template<typename T, size_t N>
    explicit Tensor(T(&arr)[N]){
        // dimension of arr, e.g. double[2][1] dimension = 2
        dimension = std::rank<T>::value + 1;
        shape.reset(new int[dimension]);
        using BaseType = typename std::remove_all_extents<T>::type();
        // type_name = typeid(BaseType).name();

        dtype_id = dtype_id_from<T>();

        getShape(arr, 0);

        // copy array to data
        total_size = 1;
        for(int i = 0; i < dimension; i++){
            total_size *= shape[i];
        }
        data_shared.reset(new VariantData[total_size]);
        data = data_shared.get();
        VariantData *pointer = data;
        copyData(arr, pointer, data_shared.get() + total_size);
    }

    Tensor(int type_id);
    Tensor();

    int cal_stride(int dim, int *shape);

    friend std::ostream &operator<<(std::ostream &os, const Tensor &t);

    template<typename T, size_t N>
    static Tensor eye(int(&size)[N]){
        Tensor t = init_with_shape(size);
        t.data_shared.reset(new VariantData[t.total_size]);
        t.data = t.data_shared.get();
        for(int i = 0; i < t.total_size; i++){
            t.data[i] = (T)0;
        }
        int min = t.shape[0] < t.shape[1] ? t.shape[0] : t.shape[1];
        for(int i = 0; i < min; i++){
            t.data[i * t.shape[1] + i] = (T)1;
        }
        return t;
    }

    Tensor operator()(int idx);
    Tensor operator()(int idx, std::pair<int, int> range);

    void operator=(const VariantData &value);

    template<size_t N>
    void operator=(const int(&arr)[N]){
        copyData(arr, data, data + total_size);
    }

    Tensor transpose(int dim1, int dim2);
    Tensor permute(int dim[]);

    template<size_t N>
    Tensor view(int(&shape)[N]){
        Tensor t = Tensor();
        t.data = data;
        t.data_shared = data_shared;
        t.total_size = total_size;
        t.dimension = N;
        t.shape.reset(new int[t.dimension]);
        for(int i = 0; i < t.dimension; i++){
            t.shape[i] = shape[i];
        }
        return t;
    }


    // Reduction operators
    // Shrink the zero dims of a tensor.
    static Tensor shrink(Tensor &ts);
    // TODO: ?
    static Tensor sum(Tensor &ts, vector<int> dims){
        for(int dim : dims){
            if(dim < 0 || dim >= ts.get_dimension()){
                throw std::invalid_argument("Invalid dimension input.");
            }
        }

        Tensor t = Tensor();
        t.shape.reset(new int[ts.get_dimension()]);

        std::sort(dims.begin(), dims.end());

        int j = 0;
        for(int i = 0; i < ts.dimension; i++){
            if(j < dims.size()){
                // Need to be added.
                if(i == dims[j]){
                    j++;
                }
            }
            else{
                break;
            }
        }
    }

    friend Tensor operator+(Tensor &t1, Tensor &t2);
};


// Math operators
static Tensor add(Tensor &t1, Tensor &t2) throw();

template<typename T, size_t N>
static Tensor rand(int(&size)[N]){
    Tensor t = Tensor::init_with_shape<T>(size, N);
    std::random_device rd;  // 获取随机数种子
    std::mt19937 gen(rd()); // 初始化Mersenne Twister伪随机数生成器
    std::uniform_real_distribution<> distrib(0, 100);

    for(int i = 0; i < t.get_total_size(); i++){
        t.data_ptr()[i] = (T)distrib(gen);
    }
    return t;
}

template<typename T>
static Tensor rand(int *arr, const int dim){
    int size[dim];
    for(int i = 0; i < dim; i++){
        size[i] = arr[i];
    }
    Tensor t = Tensor::init_with_shape<T>(size, dim);
    std::random_device rd;  // 获取随机数种子
    std::mt19937 gen(rd()); // 初始化Mersenne Twister伪随机数生成器
    std::uniform_real_distribution<> distrib(0, 100);

    for(int i = 0; i < t.get_total_size(); i++){
        t.data_ptr()[i] = (T)distrib(gen);
    }
    return t;
}

template<typename T, size_t N>
static Tensor zeros(int(&size)[N]){
    Tensor t = Tensor::init_with_shape<T>(size, N);
    for(int i = 0; i < t.get_total_size(); i++){
        t.data_ptr()[i] = (T)0;
    }
    return t;
}

template<typename T>
static Tensor zeros(int *arr, const int dim){
    int size[dim];
    for(int i = 0; i < dim; i++){
        size[i] = arr[i];
    }
    Tensor t = Tensor::init_with_shape<T>(size, dim);
    for(int i = 0; i < t.get_total_size(); i++){
        t.data_ptr()[i] = (T)0;
    }
    return t;
}

/*
    zeros_like with exact type.
*/
static Tensor zeros_like(Tensor ts){
    switch(ts.get_dtype_id()){
    case 0:
        return zeros<double>(ts.get_shape(), ts.get_dimension());
    case 1:
        return zeros<int>(ts.get_shape(), ts.get_dimension());
    case 2:
        return zeros<float>(ts.get_shape(), ts.get_dimension());
    case 3:
        return zeros<double>(ts.get_shape(), ts.get_dimension());
    }

    return Tensor();
}

template<typename T, size_t N>
static Tensor ones(int(&size)[N]){
    Tensor t = Tensor::init_with_shape(size);
    for(int i = 0; i < t.get_total_size(); i++){
        t.data_ptr()[i] = (T)1;
    }
    return t;
}

template<typename T, size_t N>
static Tensor full(int(&size)[N], T value){
    Tensor t = Tensor::init_with_shape(size);
    for(int i = 0; i < t.get_total_size(); i++){
        t.data_ptr()[i] = (T)value;
    }
    return t;
}

static Tensor cat(const std::pair<Tensor, Tensor> &tensors, int dim);

template<size_t N>
static Tensor tile(Tensor &tensor, int(&dims)[N]){
    const int dimension = tensor.get_dimension();
    if(dimension != N)
        throw std::invalid_argument("dimension of tensor and dims should be equal");
    int size[dimension];
    for(int i = 0; i < dimension; i++){
        size[i] = tensor.get_shape()[i] * dims[i];
        std::cout << size[i] << std::endl;
    }
    Tensor t = Tensor::init_with_shape(size, dimension);

    for(int i = 0; i < t.get_total_size(); i++){
        t.data_ptr()[i] = Tensor::copy_tile((tensor.data_ptr()), &t, i, tensor.get_shape(), 0);
    }
    return t;
}

static Tensor transpose(Tensor tensor, int dim1, int dim2);
static Tensor permute(Tensor tensor, int dim[]);

template<size_t N>
static Tensor view(Tensor tensor, int(&shape)[N]){
    return tensor.view(shape);
}


// Tensor operator+(Tensor &t1, Tensor &t2){
//     std::cout << "Test";
//     int *shape1 = t1.get_shape();
//     int *shape2 = t2.get_shape();
//     size_t dim1 = sizeof(shape1) / sizeof(shape1[0]);
//     size_t dim2 = sizeof(shape1) / sizeof(shape1[0]);
//     if(dim1 != dim2){
//         throw std::invalid_argument("The input tensors need to be same shape.");
//         for(int i = 0; i < dim1; i++){
//             if(shape1[i] != shape2[i]){
//                 throw std::invalid_argument("The input tensors need to be same shape.");
//             }
//         }
//     }
//     else{
//         int tid_1 = t1.type_id();
//         int tid_2 = t2.type_id();
//         int tid_rst = std::max(tid_1, tid_2);

//         Tensor result = zeros_like(t1, tid_rst);
//         auto ptr = result.data_ptr();
//         auto ptr1 = t1.data_ptr();
//         auto ptr2 = t2.data_ptr();

//         using type1 = VariantData[t1.data_ptr()[0].index()];
//         using type2 = VariantData[t2.data_ptr()[0].index()];

//         for(size_t i = 0; i < t1.get_total_size(); i++){
//             ptr[i] = ptr1[i] + ptr2[i];
//         }
//     }
// }

}

