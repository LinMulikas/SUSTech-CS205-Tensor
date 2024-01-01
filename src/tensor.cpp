#include "iostream"
#include "tensor.h"

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



        Tensor Tensor::cat(const std::pair<Tensor, Tensor>& tensors, int dim) {
            Tensor t = Tensor();
            t.dimension = tensors.first.dimension;
            t.shape.reset(new int[t.dimension]);
            for (int i = 0; i < t.dimension; i++) {
                t.shape[i] = tensors.first.shape[i];
            }
            t.shape[dim] += tensors.second.shape[dim];
            t.total_size = 1;
            for (int i = 0; i < t.dimension; i++) {
                t.total_size *= t.shape[i];
            }
            t.data_shared.reset(new VariantData[t.total_size]);
            t.data = t.data_shared.get();
            for (int i = 0; i < t.total_size; i++) {
                if (i < tensors.first.total_size) {
                    t.data[i] = tensors.first.data[i];
                } else {
                    t.data[i] = tensors.second.data[i - tensors.first.total_size];
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


        Tensor Tensor::transpose(int dim1, int dim2) {
            // 检查维度是否有效
            if (dim1 < 0 || dim1 >= dimension || dim2 < 0 || dim2 >= dimension || dim1 == dim2) {
                throw std::out_of_range("Invalid dimensions for transpose");
            }

            int old_dim1 = 1;
            for (int i = dim1 + 1; i < dimension; ++i) {
                old_dim1 *= shape[i];
            }
            int mod_dim1 = old_dim1 * shape[dim1];

            int old_dim2 = 1;
            for (int i = dim2 + 1; i < dimension; ++i) {
                old_dim2 *= shape[i];
            }
            int mod_dim2 = old_dim2 * shape[dim2];

            // 交换shape中的维度
            std::swap(shape[dim1], shape[dim2]);

            auto newData = new VariantData[total_size];

            int sub_dim1 = 1;
            for (int i = dim1 + 1; i < dimension; ++i) {
                sub_dim1 *= shape[i];
            }

            int sub_dim2 = 1;
            for (int i = dim2 + 1; i < dimension; ++i) {
                sub_dim2 *= shape[i];
            }

            // 把原来的坐标转化为新坐标，data[d1][...][d2] - data[d2][...][d1]差值计算
            for (int i = 0; i < total_size; ++i) {
                int d1 = (i % mod_dim1) / old_dim1;
                int d2 = (i % mod_dim2) / old_dim2;
                int idx = i + d1 * (sub_dim2 - old_dim1) + d2 * (sub_dim1 - old_dim2);
                newData[idx] = data[i];
            }

            for (int i = 0; i < total_size; ++i) {
                data[i] = newData[i];
            }

            return *this;
        }

        Tensor Tensor::transpose(Tensor& tensor, int dim1, int dim2) {
            return tensor.transpose(dim1, dim2);
        }


}