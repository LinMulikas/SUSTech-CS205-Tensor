#include "iostream"
#include "tensor.h"
#include <cmath>
namespace ts {
    using VariantData = std::variant<float, int, double, bool>;




         VariantData Tensor::copy_tile(VariantData *src, Tensor *dst, int idx, int *src_shape, int dim) {
            int subarrays = 1; // 当前维度的子数组数量
            for (int d = dim + 1; d < dst->dimension; ++d) {
                subarrays *= dst->shape.get()[d];
            }
            int cur_id = (idx / subarrays) % src_shape[dim];
            if (dim < dst->dimension - 1)
                return copy_tile(src + cur_id * src_shape[dim], dst, idx % subarrays, src_shape, dim + 1);
            else
                return src[idx % src_shape[dim]];
        }

        void Tensor::print(std::ostream& os, int index, int dim) const {
            if (dim == dimension) {
                // 打印单个元素
                std::visit([&os](auto&& arg) {
                    os << std::fixed << std::setprecision(4) << arg;
                }, data[index]);
            } else {
                // 打印开括号
                os << "[";

                int subarrays = 1; // 当前维度的子数组数量
                for (int d = dim + 1; d < dimension; ++d) {
                    subarrays *= shape.get()[d];
                }
                for (int i = 0; i < shape.get()[dim]; ++i) {
                    // 递归打印下一个维度
                    print(os, index + i * subarrays, dim + 1);
                    if (i < shape.get()[dim] - 1) {
                        os << ", ";
                        if (dim < dimension - 1) {  // 如果不是最内层维度，在逗号后换行
                            os << "\n" << std::string(dim + 1, ' ');  // 缩进以对齐
                        }
                    }
                }
                // 打印闭括号
                os << "]";
            }
        }


        int* Tensor::size() {
            return shape.get();
        }

        std::string Tensor::type() {
            return type_name;
        }

        VariantData* Tensor::data_ptr() {
            return data;
        }

        int Tensor::get_total_size() {
            return total_size;
        }

        Tensor::Tensor() {
            dimension = 0;
            shape = nullptr;
            data = nullptr;
            total_size = 0;
        }

        std::ostream& operator<<(std::ostream& os, const Tensor& t) {
            if (t.dimension == 0 || t.total_size == 0) {
                os << "[]";
                return os;
            }

            t.print(os, 0, 0);

            return os;
        }


        int Tensor::get_dimension() const {
            return dimension;
        }

        int* Tensor::get_shape() const {
            return shape.get();
        }

        Tensor cat(std::pair<Tensor, Tensor>& tensors, int dim) {
            const int dimension = tensors.first.get_dimension();
            int size[dimension];
            for (int i = 0; i < dimension; i++) {
                size[i] = tensors.first.get_shape()[i];
            }
            size[dim] += tensors.second.get_shape()[dim];
            Tensor t = Tensor::getShapeTensor(size, dimension);
            for (int i = 0; i < t.get_total_size(); i++) {
                if (i < tensors.first.get_total_size())
                    t.data_ptr()[i] = tensors.first.data_ptr()[i];
                else {
                    t.data_ptr()[i] = tensors.second.data_ptr()[i - tensors.first.get_total_size()];
                }
            }
            return t;
        }


        Tensor Tensor::operator()(int idx) {
            Tensor t = Tensor();
            t.dimension = dimension - 1;
            t.shape.reset(new int[t.dimension]);
            for (int i = 0; i < t.dimension; i++) {
                t.shape[i] = shape[i + 1];
            }
            t.total_size = total_size / shape[0];
            t.data = shape[0] * idx + data;
            return t;
        }


        Tensor Tensor::operator()(int idx, std::pair<int, int> range) {
            Tensor t = Tensor();
            t.dimension = dimension - 1;
            t.shape.reset(new int[t.dimension]);
            t.shape[0] = range.second - range.first;
            t.total_size = t.shape[0];
            for (int i = 1; i < t.dimension; i++) {
                t.shape[i] = shape[i + 1];
                t.total_size *= shape[i + 1];
            }
            ;
            t.data = shape[0] * idx + data + range.first;
            return t;
        }

        void Tensor::operator=(const VariantData& value) {
            for (int i = 0; i < total_size; i++) {
                data[i] = value;
            }
        }

        int Tensor::cal_stride(int dim, int *shape) {
            int stride = 1;
            for (int i = dim + 1; i < dimension; i++) {
                stride *= shape[i];
            }
            return stride;
        }


        Tensor Tensor::transpose(int dim1, int dim2) {
            // 检查维度是否有效
            if (dim1 < 0 || dim1 >= dimension || dim2 < 0 || dim2 >= dimension || dim1 == dim2) {
                throw std::out_of_range("Invalid dimensions for transpose");
            }

            int old_dim1 = cal_stride(dim1, shape.get());
            int mod_dim1 = old_dim1 * shape[dim1];

            int old_dim2 = cal_stride(dim2, shape.get());
            int mod_dim2 = old_dim2 * shape[dim2];

            // 交换shape中的维度
            int *newShape = new int[dimension];
            for (int i = 0; i < dimension; ++i) {
                newShape[i] = shape[i];
            }
            newShape[dim1] = shape[dim2];
            newShape[dim2] = shape[dim1];

            auto newData = new VariantData[total_size];

            int sub_dim1 = cal_stride(dim1, newShape);

            int sub_dim2 = cal_stride(dim2, newShape);
            // 把原来的坐标转化为新坐标，data[d1][...][d2] - data[d2][...][d1]差值计算
            for (int i = 0; i < total_size; ++i) {
                int d1 = (i % mod_dim1) / old_dim1;
                int d2 = (i % mod_dim2) / old_dim2;
                int idx = i + d1 * (sub_dim2 - old_dim1) + d2 * (sub_dim1 - old_dim2);
                newData[idx] = data[i];
            }

            Tensor tensor = Tensor::getShapeTensor(newShape, dimension);
            for (int i = 0; i < total_size; ++i) {
                tensor.data_ptr()[i] = newData[i];
            }

            return tensor;
        }

        //新加的
       

        
        Tensor  Tensor::add(Tensor t1)
        {
            is_size_ok_pointwise(t1);
            VariantData* new_data=new VariantData[total_size];
            for(int i=0;i<total_size;i++)
            {
                  new_data[i]=data[i]+t1.data[i];
            }
            t1.data=new_data;
            
            return t1; 
        }
        Tensor Tensor::sub(Tensor t1)
        {
            is_size_ok_pointwise(t1);
            VariantData* new_data=new VariantData[total_size];
            for(int i=0;i<total_size;i++)
            {
                  new_data[i]=data[i]-t1.data[i];
            }
            this->data=new_data;
            
            return *(this); 

        }
        Tensor Tensor::mul(Tensor t1){
              is_size_ok_pointwise(t1);
            VariantData* new_data=new VariantData[total_size];
            for(int i=0;i<total_size;i++)
            {
                  new_data[i]=data[i]*t1.data[i];
            }
            this->data=new_data;
            
            return *(this); 
        }
        Tensor Tensor::div(Tensor t1){
            is_size_ok_pointwise(t1);
            for (int i=0;i<total_size;i++)
            {
                if(t1.data[i]==0) throw std::invalid_argument("invalid divisor");
            }
            VariantData* new_data=new VariantData[total_size];
            for(int i=0;i<total_size;i++)
            {
                  new_data[i]=data[i]/t1.data[i];
            }
            this->data=new_data;
            
            return *(this); 
        }

        Tensor Tensor::log(){
            for(int i=0;i<total_size;i++)
            {
                if(data[i]<=0) throw std::invalid_argument("invalid argument");
            }
            for(int i=0;i<total_size;i++)
            {
                data[i]=std::log(data[i]);
            }
        }
















        int calculateIndex(int* indices, int* strides, int dimension) {
            int index = 0;
            for (int i = 0; i < dimension; ++i) {
                index += strides[i] * indices[i];
            }
            return index;
        }

        int* getStrides(int* shape, int dimension) {
            int* strides = new int[dimension];
            for (int i = 0; i < dimension; ++i) {
                strides[i] = 1;
                for (int j = i + 1; j < dimension; ++j) {
                    strides[i] *= shape[j];
                }
            }
            return strides;
        }

        Tensor Tensor::permute(int dim[]) {
            int *strides = getStrides(shape.get(), dimension);
            std::shared_ptr<int> newShape(new int[dimension]);

            for (int i = 0; i < dimension; ++i) {
                newShape.get()[i] = shape[dim[i]];
            }
            int *newStrides = getStrides(newShape.get(), dimension);
            auto newArr = new VariantData[total_size];

            for (int i = 0; i < total_size; ++i) {
                int* indices = new int[dimension];
                int idx = i;
                for (int j = 0; j < dimension; ++j) {
                    indices[j] = idx / strides[j];
                    idx = idx % strides[j];
                }
                int* permuted_indices = new int[dimension];
                for (int j = 0; j < dimension; ++j) {
                    permuted_indices[j] = indices[dim[j]];
                }

                int newIdx = calculateIndex(permuted_indices, newStrides, dimension);
                newArr[newIdx] = data[i];
                delete[] indices;
                delete[] permuted_indices;
            }

            Tensor t = Tensor::getShapeTensor(newShape.get(), dimension);
            for (int i = 0; i < total_size; ++i) {
                t.data_ptr()[i] = newArr[i];
            }

            delete[] strides;
            delete[] newStrides;
            return t;
        }

        Tensor transpose(Tensor tensor, int dim1, int dim2) {
            return tensor.transpose(dim1, dim2);
        }

        Tensor permute(Tensor tensor, int dim[]) {
            return tensor.permute(dim);
        }



    //新加的

        Tensor add(Tensor t1,Tensor t2){
              return t1.add(t2);
        }
        Tensor sub(Tensor t1, Tensor t2){
            return t1.sub(t2);
        }
        Tensor mul(Tensor t1,Tensor t2){
            return t1.mul(t2);
        }
        Tensor div(Tensor t1,Tensor t2){
            return t1.div(t2);
        }





















}