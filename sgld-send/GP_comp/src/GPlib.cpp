
/*
 * =====================================================================================
 *
 *       Filename:  GPLib.cpp
 *
 *    Description:  Gaussian Prosess Library
 *
 *        Version:  1.0
 *        Created:  01/25/2018 15:39:11
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Jian Kang (JK), jiankang@umich.edu
 *   Organization:  University of Michigan
 *
 * =====================================================================================
 */

# include <cstdlib>
# include <cmath>
# include <iostream>
# include <sstream>
# include <iomanip>
# include <ctime>
# include <cstring>
# include <list>
# include <vector>
# include <unordered_set>
using namespace std;
# include "hermite_polynomial.h"
# include <pybind11/pybind11.h>
# include <pybind11/numpy.h>

namespace py = pybind11;

double nchoosek(double n, double k){
  return(rint(exp(lgamma(n+1.0)-lgamma(k+1.0)-lgamma(n-k+1.0))));
}

template <class commonType>
void vector_copy(commonType* out, commonType* in, int size){
   for(int i=0;i<size; i++ ){
      out[i] = in[i];
   }
}


template <class commonType>
void vector_copy(commonType* out,int out_start, commonType* in,int in_start, int size){
   for(int i=0;i<size; i++ ){
      out[i+out_start] = in[i+in_start];
   }
}

int* xsimplex(int p, int n, int& count){
   int target = 1;
   int i = 0;
   count = (int)nchoosek(n+p-1,n);
   int* out = new int [p*count];
   if(p==1){
     out[0] = n;
     return out;
   }
   int* x = new int[p];
   int p1 = p-1;
   x[0] = n;
   for(int i=1;i<p;i++){
      x[i] = 0;
   }
   while(true){
      vector_copy<int>(out,i*p,x,0,p);
      i++;
      x[target-1]--;
      if(target<p1){
          target++;
          x[target-1] = 1+x[p1];
          x[p1] = 0;
      }
      else{
         x[p1]++;
         while(x[target-1]==0){
            target--;
            if(target==0){
               vector_copy<int>(out,i*p,x,0,p);
               i++;
               delete [] x;
               return out;
            }

         }
      }
   }
   return out;
}

int* GP_xsimplex(int d, int n, int* xsimplex_end){
   int total_size = (int)nchoosek(n+d,d);
   int* all_xsimplex = new int[d*total_size];
   xsimplex_end[0] = 1;
   for(int i=0; i<d;i++)
      all_xsimplex[i] = 0;

   for(int i=1;i<n+1;i++){
     int temp_size = 0.0;
     int* temp = xsimplex(d,i,temp_size);
     xsimplex_end[i] = xsimplex_end[i-1]+temp_size;
     vector_copy<int>(all_xsimplex,d*xsimplex_end[i-1],temp,0,temp_size*d);
     delete[] temp;
   }
   return all_xsimplex;

}


/*template<class commonType>
void print_array(commonType* array, int nrow, int ncol,int start=0){
   for(int i=0;i<nrow;i++){
      for(int j=0;j<ncol-1;j++){
         std::cout << array[start+i+j*nrow] << " ";
      }
      std::cout << array[start+i+(ncol-1)*nrow] << std::endl;
   }
}*/

template<class commonType>
commonType* expand_grid(commonType* grid_x,int size_x,commonType* grid_y,int size_y,commonType* grid_z,int size_z, int& grids_size){
   grids_size = size_x*size_y*size_z;
   commonType* grids = new commonType[grids_size*3];
   for(int i=0;i<size_x;i++){
    for(int j=0;j<size_y;j++){
      for(int k=0;k<size_z;k++){
         int loc = size_y*(size_x*k+j)+i;
         grids[loc] = grid_x[i];
         grids[grids_size + loc] = grid_y[j];
         grids[2*grids_size + loc] = grid_z[k];
      }
    }
   }
   return grids;
}

template<class commonType>
commonType* expand_grid(commonType* grid_x,int size_x,commonType* grid_y,int size_y,int& grids_size){
   grids_size = size_x*size_y;
   commonType* grids = new commonType[size_x*size_y*2];
   for(int i=0;i<size_x;i++){
      for(int j=0;j<size_y;j++){
         grids[size_y*j+i] = grid_x[i];
         grids[size_y*(size_x+j)+i] = grid_y[j];
      }
   }
   return grids;
}




template<class commonType>
commonType* seq(commonType start, commonType end, int length){
   commonType* x = new commonType[length];
   for(int j=0;j<length;j++){
       x[j] = start+(end-start)*(j/(length-1.0));
   }
   return x;
}

struct is_near {
  bool operator() (double first, double second)
  { return (fabs(first-second)<1e-5); }
};

template<class commonType>
commonType* unique_slow(commonType* x,int length,int& out_length,int start=0){
  std::list<commonType> mylist (x+start,x+start+length);
  mylist.sort();
  mylist.unique();
  out_length = mylist.size();
  commonType* uni_x = new commonType[out_length];
  int i = 0;
  typename std::list<commonType>::iterator it;
  for (it=mylist.begin(); it!=mylist.end(); ++it)
    uni_x[i++] = *it;
  return uni_x;
}

template<class commonType>
commonType* unique(commonType* x,int length,int& out_length,int start=0){
  std::unordered_set<commonType> myset (x+start,x+start+length);
  out_length = myset.size();
  commonType* uni_x = new commonType[out_length];
  int i = 0;
  typename std::unordered_set<commonType>::iterator it;
  for (it=myset.begin(); it!=myset.end(); ++it)
    uni_x[i++] = *it;
  return uni_x;
}

template<class commonType>
int* which(commonType* x, int length, commonType value, int& out_size){
   std::vector<int> vec;
   for(int i=0;i<length;i++){
      if(x[i]==value){
         vec.push_back(i);
      }
   }
   out_size = vec.size();
   int * idx;
   if(vec.size()>0){
   idx = new int[vec.size()];
   int i = 0;
   for(std::vector<int>::iterator it=vec.begin(); it!=vec.end();++it){
      idx[i++] = *it;
   }
   }
   else{
    idx = NULL;
   }
   return idx;
}

template<class commonType>
int* match(commonType* x, int length, commonType* unique_values, int num_values){
   int* idx = new int[length];
   for(int i=0; i<num_values; i++){
     for(int j=0; j<length; j++){
        if(unique_values[i]==x[j]){
           idx[j] = i;
        }
     }
   }
   return idx;
}

double* GP_eigen_funcs_comp(double* uqgrid,int uqgrid_size,int* uqidx,int dim,int grids_size,
                            int* xsimplex_list,int* xsimplex_list_end,int poly_degree,double cn){

   double* eigen_funcs  = NULL;
   double sqrt2c = sqrt(2.0*cn);
   double D = pow(sqrt2c,0.5*dim);

   double* exp_neg_c_uqgrid_sq = new double[uqgrid_size];
   double* sqrt2c_uqgrid = new double[uqgrid_size];
   for(int i=0;i<uqgrid_size;i++){
      exp_neg_c_uqgrid_sq[i] = exp(-cn*uqgrid[i]*uqgrid[i]);
      sqrt2c_uqgrid[i] = sqrt2c*uqgrid[i];
   }

   double* uqhermite =  hn_polynomial_value(uqgrid_size,poly_degree,sqrt2c_uqgrid);

   int funcs_num = xsimplex_list_end[poly_degree];
   eigen_funcs = new double[grids_size*funcs_num];

   int poly_degree_1 = poly_degree+1;
   for(int i=0;i<grids_size;i++){
     for(int k=0; k<poly_degree_1; k++){
      int start, end;
      if(k==0){
         start = 0;
      }
      else{
         start = xsimplex_list_end[k-1];
      }
      end = xsimplex_list_end[k]-1;
      for(int j=start;j<=end; j++){
         long eigen_loc = i+grids_size*j;
         eigen_funcs[eigen_loc] = D;
         for(int l=0;l<dim;l++){
            int degree = xsimplex_list[dim*j+l];
            int x_loc = uqidx[grids_size*l+i];
            eigen_funcs[eigen_loc] *= exp_neg_c_uqgrid_sq[x_loc]*uqhermite[degree*uqgrid_size+x_loc];
         }
      }

     }
   }

   delete[] uqhermite;
   delete[] exp_neg_c_uqgrid_sq;
   delete[] sqrt2c_uqgrid;
   return eigen_funcs;
}

void R_GP_eigen_funcs_comp(double* eigen_funcs, double* uqgrid,int uqgrid_size,int* uqidx,int dim,int grids_size,
                            int* xsimplex_list,int* xsimplex_list_end,int poly_degree,double cn){

   double sqrt2c = sqrt(2.0*cn);
   double D = pow(sqrt2c,0.5*dim);

   double* exp_neg_c_uqgrid_sq = new double[uqgrid_size];
   double* sqrt2c_uqgrid = new double[uqgrid_size];
   for(int i=0;i<uqgrid_size;i++){
      exp_neg_c_uqgrid_sq[i] = exp(-cn*uqgrid[i]*uqgrid[i]);
      sqrt2c_uqgrid[i] = sqrt2c*uqgrid[i];
   }

   double* uqhermite =  hn_polynomial_value(uqgrid_size,poly_degree,sqrt2c_uqgrid);

 //  int funcs_num = xsimplex_list_end[poly_degree];

   int poly_degree_1 = poly_degree+1;
   for(int i=0;i<grids_size;i++){
     for(int k=0; k<poly_degree_1; k++){
      int start, end;
      if(k==0){
         start = 0;
      }
      else{
         start = xsimplex_list_end[k-1];
      }
      end = xsimplex_list_end[k]-1;
      for(int j=start;j<=end; j++){
         long eigen_loc = i+grids_size*j;
         eigen_funcs[eigen_loc] = D;
         for(int l=0;l<dim;l++){
            int degree = xsimplex_list[dim*j+l];
            int x_loc = uqidx[grids_size*l+i];
            eigen_funcs[eigen_loc] *= exp_neg_c_uqgrid_sq[x_loc]*uqhermite[degree*uqgrid_size+x_loc];
         }
      }

     }
   }

   delete[] uqhermite;
   delete[] exp_neg_c_uqgrid_sq;
   delete[] sqrt2c_uqgrid;
}


double inner_prod(double* v, double* u, long n, long v_start = 0, long u_start = 0){
   double y = 0.0;
   for(int i=0; i<n; i++){
      y += v[i+v_start]*u[i+u_start];
   }
   return y;
}

void proj_v_on_u(double* v, double* u, long n, long v_start = 0, long u_start = 0){
   double uv = inner_prod(v,u,n,v_start,u_start);
   double uu = inner_prod(u,u,n,u_start,u_start);
   if(uu>0.0){
      uv /= uu;
      for(int i=0;i<n;i++){
         v[v_start+i] = uv*u[u_start+i];
      }
   } else{
      for(int i=0;i<n;i++){
         v[v_start+i] = 0.0;
      }
   }
}

void proj_v_on_e(double* v, double* e, long n, long v_start = 0, long e_start = 0){
   double ev = inner_prod(v,e,n,v_start,e_start);
   for(int i=0;i<n;i++){
      v[v_start+i] = ev*e[e_start+i];
   }
}

void normalize_vec(double* u, long n, long u_start){
   double uu = inner_prod(u,u,n,u_start,u_start);
   if(uu>0.0){
      double sqrt_uu = sqrt(uu);
      for(int i=0;i<n;i++){
         u[u_start+i] /= sqrt_uu;
      }
   }
}


void R_GP_eigen_funcs_orth_comp(double* eigen_funcs, double* uqgrid, int uqgrid_size,int* uqidx,int dim,int grids_size,
                           int* xsimplex_list,int* xsimplex_list_end,int poly_degree,double cn){

   double sqrt2c = sqrt(2.0*cn);
   double D = pow(sqrt2c,0.5*dim);

   double* exp_neg_c_uqgrid_sq = new double[uqgrid_size];
   double* sqrt2c_uqgrid = new double[uqgrid_size];
   double* temp_vec = new double[grids_size];
   double* proj_vec = new double[grids_size];

   for(int i=0;i<uqgrid_size;i++){
      exp_neg_c_uqgrid_sq[i] = exp(-cn*uqgrid[i]*uqgrid[i]);
      sqrt2c_uqgrid[i] = sqrt2c*uqgrid[i];
   }

   double* uqhermite =  hn_polynomial_value(uqgrid_size,poly_degree,sqrt2c_uqgrid);

   int poly_degree_1 = poly_degree+1;
   for(int k=0; k<poly_degree_1; k++){

         int start, end;
         if(k==0){
            start = 0;
         }
         else{
            start = xsimplex_list_end[k-1];
         }
         end = xsimplex_list_end[k]-1;
         for(int j=start;j<=end; j++){

            for(int i=0;i<grids_size;i++){
               //eigen_funcs[eigen_loc] = D;
               temp_vec[i] = D;
               for(int l=0;l<dim;l++){
                  int degree = xsimplex_list[dim*j+l];
                  int x_loc = uqidx[grids_size*l+i];
                  //eigen_funcs[eigen_loc] *= exp_neg_c_uqgrid_sq[x_loc]*uqhermite[degree*uqgrid_size+x_loc];
                  temp_vec[i] *= exp_neg_c_uqgrid_sq[x_loc]*uqhermite[degree*uqgrid_size+x_loc];
               }
            }

            for(int i=0; i<grids_size;i++){
               long eigen_loc = i+grids_size*j;
               eigen_funcs[eigen_loc] = temp_vec[i];
            }
            if(j>0){
               for(int m=0;m<j;m++){
                  for(int i=0;i<grids_size;i++){
                     proj_vec[i] = temp_vec[i];
                  }
                  proj_v_on_u(proj_vec,eigen_funcs,grids_size,0,grids_size*m);
                  for(int i=0; i<grids_size; i++){
                     eigen_funcs[i+grids_size*j] -= proj_vec[i];
                  }
               }
            }
            //normalize_vec(eigen_funcs,grids_size,grids_size*j);

         }
   }

   delete[] uqhermite;
   delete[] exp_neg_c_uqgrid_sq;
   delete[] sqrt2c_uqgrid;
   delete[] temp_vec;
   delete[] proj_vec;
}

void R_GP_eigen_funcs(double* eigen_funcs, double* grids, int grids_size,int dim, int poly_degree, double a, double b){
  double cn = sqrt(a*a+2*a*b);
  int uqgrid_size = 0;
  double* uqgrid = unique<double>(grids,dim*grids_size,uqgrid_size);
  int* uqidx = match<double>(grids,dim*grids_size,uqgrid,uqgrid_size);
  int* xsimplex_list_end = new int [poly_degree+1];
  int* xsimplex_list = GP_xsimplex(dim,poly_degree,xsimplex_list_end);

  R_GP_eigen_funcs_comp(eigen_funcs,uqgrid,uqgrid_size,uqidx,dim,grids_size,
                            xsimplex_list, xsimplex_list_end,poly_degree,cn);

  delete[] uqgrid;
  delete[] uqidx;
  delete[] xsimplex_list_end;
  delete[] xsimplex_list;
}

void R_GP_eigen_funcs_orth(double* eigen_funcs, double* grids, int grids_size,int dim, int poly_degree, double a, double b){
   double cn = sqrt(a*a+2*a*b);
   int uqgrid_size = 0;
   double* uqgrid = unique<double>(grids,dim*grids_size,uqgrid_size);
   int* uqidx = match<double>(grids,dim*grids_size,uqgrid,uqgrid_size);
   int* xsimplex_list_end = new int [poly_degree+1];
   int* xsimplex_list = GP_xsimplex(dim,poly_degree,xsimplex_list_end);

   R_GP_eigen_funcs_orth_comp(eigen_funcs,uqgrid,uqgrid_size,uqidx,dim,grids_size,
                         xsimplex_list, xsimplex_list_end,poly_degree,cn);

   delete[] uqgrid;
   delete[] uqidx;
   delete[] xsimplex_list_end;
   delete[] xsimplex_list;
}


double* GP_eigen_funcs(double* grids, int grids_size,int dim, int poly_degree, double a, double b, int& num_funcs){
  double cn = sqrt(a*a+2*a*b);
  int uqgrid_size = 0;
  double* uqgrid = unique<double>(grids,dim*grids_size,uqgrid_size);
  int* uqidx = match<double>(grids,dim*grids_size,uqgrid,uqgrid_size);
  int* xsimplex_list_end = new int [poly_degree+1];
  int* xsimplex_list = GP_xsimplex(dim,poly_degree,xsimplex_list_end);
  num_funcs = xsimplex_list_end[poly_degree];
  double* eigen_funcs = GP_eigen_funcs_comp(uqgrid,uqgrid_size,uqidx,dim,grids_size,
                            xsimplex_list, xsimplex_list_end,poly_degree,cn);

  delete[] uqgrid;
  delete[] uqidx;
  delete[] xsimplex_list_end;
  delete[] xsimplex_list;
  return eigen_funcs;
}

template<class commonType>
commonType* matmul_AtA(commonType* A,int nrow,int ncol){
   commonType* AtA = new commonType[ncol*ncol];

   for(int i=0;i<ncol;i++){
     AtA[i*ncol+i] = 0.0;
     for(int k=0;k<nrow;k++){
       AtA[i*ncol+i] += A[i*nrow+k]*A[i*nrow+k];
     }
   }

   for(int i=0; i<(ncol-1); i++){
     for(int j=i+1;j<ncol;j++){
       AtA[i*ncol+j] = 0.0;
        for(int k=0;k<nrow;k++){
          AtA[i*ncol+j] += A[i*nrow+k]*A[j*nrow+k];
        }
       AtA[j*ncol+i] = AtA[i*ncol+j];
     }
   }
   return AtA;
}

py::array_t<double> py_GP_eigen_funcs(py::array_t<double> eigen_funcs, py::array_t<double> grids, int* grids_size, int* dim, int* poly_degree, double* a, double* b) {
    auto buf1 = eigen_funcs.request();
    auto buf2 = grids.request();

    double* ptr1 = static_cast<double*>(buf1.ptr);
    double* ptr2 = static_cast<double*>(buf2.ptr);
    
    R_GP_eigen_funcs(ptr1, ptr2, *grids_size, *dim, *poly_degree, *a, *b);
    
    py::array_t<double> result(buf1.size);
    auto result_buf = result.request();
    double* result_ptr = static_cast<double*>(result_buf.ptr);
    std::memcpy(result_ptr, ptr1, buf1.size * sizeof(double));

    return result;
}

py::array_t<double> py_GP_eigen_funcs_orth(py::array_t<double> eigen_funcs, py::array_t<double> grids, int* grids_size, int* dim, int* poly_degree, double* a, double* b) {
    auto buf1 = eigen_funcs.request();
    auto buf2 = grids.request();

    double* ptr1 = static_cast<double*>(buf1.ptr);
    double* ptr2 = static_cast<double*>(buf2.ptr);
    
    R_GP_eigen_funcs_orth(ptr1, ptr2, *grids_size, *dim, *poly_degree, *a, *b);
    
    py::array_t<double> result(buf1.size);
    auto result_buf = result.request();
    double* result_ptr = static_cast<double*>(result_buf.ptr);
    std::memcpy(result_ptr, ptr1, buf1.size * sizeof(double));

    return result;
}

PYBIND11_MODULE(GPlib, m) {
    m.def("GP_eigen_funcs", &py_GP_eigen_funcs, "Python wrapper for GP_eigen_funcs");
    m.def("GP_eigen_funcs_orth", &py_GP_eigen_funcs_orth, "Python wrapper for GP_eigen_funcs_orth");
}