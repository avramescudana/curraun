#include "npy.hpp"
#include <vector>
#include <string>

#include <iostream>
using namespace std;

int main() {

// Path to the file containing the numpy array
const std::string path {"uplus_glasma.npy"};

// Read the glasma numpy array from the file in the given path
npy::npy_data d = npy::read_npy<std::complex<double>>(path);

// Get the data values and shape of the numpy array
// The data is a 1D array of size (nxplus * ny * nz * nc) with complex values
std::vector<std::complex<double>> data = d.data;
// The shape is a vector of size 4, with the dimensions of the numpy array [nxplus, ny, nz, nc]
std::vector<unsigned long> shape = d.shape;

// Print the shape of the numpy array 
//   for (int j = 0; j < shape.size(); ++j) {
//         std::cout << shape[j] << std::endl;
//     }

// 4D array, all time slices
// Array of size (nxplus, ny, nz, nc) where nc=9
// Current index is ixplus * ny * nz * nc + iy * nz * nc + iz * nc + ic

    // Number of time steps in x^+, can also be read from the file `parameters_glasma.txt`
    int nxplus = shape[0];
    // Number of lattice points in y, can also be read from the file `parameters_glasma.txt`
    int ny = shape[1];
    // Number of lattice points in z, the same as ny
    int nz = shape[2];
    // Number of matrix color elements, 9 for SU(3) in the fundemental representation
    int nc = shape[3];

    // Current index in each x^+, y, z slice
    // These test values are used to test that the array is correctly read from file
    int ixplus = 10;
    int iy = 12;
    int iz = 10;

    // Current index in the 4D array
    int xplus_y_z_index = ixplus * ny * nz * nc + iy * nz * nc + iz * nc;

    // Loop through the matrix elements in the current x^+, y, z slice
    for (int ic= 0; ic < nc; ++ic) {
        // Get the color component index
        int color_index = xplus_y_z_index  + ic;
        // Print the color component
        std::cout << data[color_index] << std::endl;
    }

}