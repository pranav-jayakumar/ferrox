use pyo3::{Python, pyfunction, pymodule, wrap_pyfunction, PyResult};
use pyo3::types::PyModule;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;



fn chebyshev_sin(mut x: f64) -> f64 {
    // Constants
    const PI: f64 = 3.14159265358979323846;
    const HALF_PI: f64 = PI / 2.0;
    const TWO_PI: f64 = 2.0 * PI;

    // Chebyshev coefficients for sine approximation
    const C1: f64 = -0.16666667;
    const C2: f64 = 0.00833333;
    const C3: f64 = -0.00019841;
    const C4: f64 = 2.7526e-06;

    // Reduce x to range [0, 2 * pi]
    x = x % TWO_PI;

    // Use symmetry to reduce the range to [0, pi/2]
    let sign: f64 = if x > PI { -1.0 } else { 1.0 };
    x = if x > PI { x - PI } else { x };
    x = if x > HALF_PI { PI - x } else { x };

    // Apply Chebyshev polynomial approximation for sin(x) in [0, pi/2]
    let x2: f64 = x * x;
    x * (1.0 + C1 * x2 + C2 * x2 * x2 + C3 * x2 * x2 * x2 + C4 * x2 * x2 * x2 * x2) * sign
}

fn newton_sqrt(x: f64) -> f64 {
    let mut guess: f64 = x / 2.0;
    for _ in 0..10 {
        guess = guess + x / guess
    }
    guess
}


/// SIMD-based dot product for x86_64 (SSE/AVX intrinsics)
#[cfg(target_arch = "x86_64")]
unsafe fn dot_product_x86(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = _mm_setzero_ps();
    
    for i in (0..a.len()).step_by(4) {
        let va = _mm_loadu_ps(&a[i]);
        let vb = _mm_loadu_ps(&b[i]);
        sum = _mm_add_ps(sum, _mm_mul_ps(va, vb));
    }

    // Sum all elements of the SIMD vector
    let sum_arr: [f32; 4] = std::mem::transmute(sum);
    sum_arr.iter().sum()
}

/// Generic SIMD-based dot product function with conditional compilation
pub fn dot_product_fe(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        panic!("Vectors must have the same length");
    }

    // Use architecture-specific SIMD implementation
    #[cfg(target_arch = "x86_64")]
    {
        unsafe { dot_product_x86(a, b) }
    }

    #[cfg(target_arch = "aarch64")]
    {
        unsafe { dot_product_arm(a, b) }
    }

    // Fallback for unsupported architectures
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        panic!("Error: Unsupported Architecture");
    }
}


/// SIMD-based dot product for ARM (NEON intrinsics)
#[cfg(target_arch = "aarch64")]
unsafe fn dot_product_arm(a: &[f32], b: &[f32]) -> f32 {
    let mut sum: float32x4_t = vdupq_n_f32(0.0);
    
    for i in (0..a.len()).step_by(4) {
        let va: float32x4_t = vld1q_f32(&a[i]);
        let vb: float32x4_t = vld1q_f32(&b[i]);
        sum = vmlaq_f32(sum, va, vb);
    }

    // Sum all elements of the SIMD vector
    let sum_array: [f32; 4] = std::mem::transmute(sum);
    sum_array.iter().sum()
}

/// SIMD-optimized traditional matrix multiplication for x86_64 (AVX2)
#[cfg(target_arch = "x86_64")]
unsafe fn simd_traditional_mult_x86(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = a.len();
    let m = b[0].len();
    let p = b.len();
    
    // Initialize result matrix with zeros
    let mut result = vec![vec![0.0; m]; n];
    
    // Perform matrix multiplication using SIMD
    for i in 0..n {
        for j in 0..m {
            let mut sum = _mm256_setzero_pd();  // Initialize SIMD sum for AVX2
            
            // Process 4 elements at a time using SIMD (for 256-bit AVX2)
            for k in (0..p).step_by(4) {
                let va = _mm256_loadu_pd(&a[i][k]);    // Load 4 elements from row i of matrix a
                let vb = _mm256_loadu_pd(&b[k][j]);    // Load 4 elements from column j of matrix b
                sum = _mm256_fmadd_pd(va, vb, sum);    // Multiply and add (fused multiply-add)
            }

            // Store the result into the final matrix
            let sum_arr: [f64; 4] = std::mem::transmute(sum);  // Convert SIMD register to array
            result[i][j] = sum_arr.iter().sum();  // Sum the four SIMD results to a scalar
        }
    }
    result
}

/// SIMD-optimized traditional matrix multiplication for ARM (NEON)
#[cfg(target_arch = "aarch64")]
unsafe fn simd_traditional_mult_arm(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n: usize = a.len();
    let m: usize = b[0].len();
    let p: usize = b.len();
    
    // Initialize result matrix with zeros
    let mut result: Vec<Vec<f64>> = vec![vec![0.0; m]; n];
    
    // Perform matrix multiplication using SIMD
    for i in 0..n {
        for j in 0..m {
            let mut sum: float64x2_t = vdupq_n_f64(0.0);  // Initialize SIMD sum for NEON
            
            // Process 2 elements at a time using SIMD (for 128-bit NEON)
            for k in (0..p).step_by(2) {
                let va: float64x2_t = vld1q_f64(&a[i][k]);  // Load 2 elements from row i of matrix a
                let vb: float64x2_t = vld1q_f64(&b[k][j]);  // Load 2 elements from column j of matrix b
                sum = vfmaq_f64(sum, va, vb);  // Multiply and accumulate (fused multiply-add)
            }

            // Store the result into the final matrix
            let sum_arr: [f64; 2] = std::mem::transmute(sum);  // Convert SIMD register to array
            result[i][j] = sum_arr.iter().sum();  // Sum the SIMD results to a scalar
        }
    }
    result
}

fn traditional_mult(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    // Use architecture-specific SIMD implementation
    #[cfg(target_arch = "x86_64")]
    {
        unsafe { simd_traditional_mult_x86(a, b) }
    }

    #[cfg(target_arch = "aarch64")]
    {
        unsafe { simd_traditional_mult_arm(a, b) }
    }

    // Fallback for unsupported architectures
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        panic!("Error: Unsupported Architecture");
    }
}

/// Calculate the next power of two greater than or equal to `n`
fn next_power_of_two(n: usize) -> usize {
    if n == 0 {
        return 1;
    }
    let mut power: usize = 1;
    while power < n {
        power <<= 1;
    }
    power
}

/// Pads a matrix to the given rows and columns size with zeroes
fn pad_matrix(matrix: &[Vec<f64>], rows: usize, cols: usize) -> Vec<Vec<f64>> {
    let original_rows: usize = matrix.len();
    let original_cols: usize = matrix[0].len();
    
    let mut padded_matrix: Vec<Vec<f64>> = vec![vec![0.0; cols]; rows];
    for i in 0..original_rows {
        for j in 0..original_cols {
            padded_matrix[i][j] = matrix[i][j];
        }
    }
    padded_matrix
}

/// Splits a matrix into four submatrices (quadrants)
fn split_matrix(matrix: &[Vec<f64>]) -> (Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let n: usize = matrix.len();
    let half_size: usize = n / 2;

    // Create four new submatrices
    let mut a11: Vec<Vec<f64>> = vec![vec![0.0; half_size]; half_size];
    let mut a12: Vec<Vec<f64>> = vec![vec![0.0; half_size]; half_size];
    let mut a21: Vec<Vec<f64>> = vec![vec![0.0; half_size]; half_size];
    let mut a22: Vec<Vec<f64>> = vec![vec![0.0; half_size]; half_size];

    // Fill submatrices by splitting the original matrix into quadrants
    for i in 0..half_size {
        for j in 0..half_size {
            a11[i][j] = matrix[i][j];
            a12[i][j] = matrix[i][j + half_size];
            a21[i][j] = matrix[i + half_size][j];
            a22[i][j] = matrix[i + half_size][j + half_size];
        }
    }

    (a11, a12, a21, a22)
}

/// Unpads a padded matrix back to its original size
fn unpad_matrix(matrix: Vec<Vec<f64>>, rows: usize, cols: usize) -> Vec<Vec<f64>> {
    matrix
        .into_iter()
        .take(rows)
        .map(|row: Vec<f64>| row.into_iter().take(cols).collect())
        .collect()
}

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// SIMD-optimized matrix addition for x86_64 (AVX)
#[cfg(target_arch = "x86_64")]
unsafe fn simd_add_matrix_x86(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = a.len();
    let mut result = vec![vec![0.0; n]; n];

    for i in 0..n {
        for j in (0..n).step_by(4) {
            let va = _mm256_loadu_pd(&a[i][j]);  // Load 4 elements from matrix a
            let vb = _mm256_loadu_pd(&b[i][j]);  // Load 4 elements from matrix b
            let vsum = _mm256_add_pd(va, vb);    // Perform addition
            _mm256_storeu_pd(&mut result[i][j], vsum);  // Store result
        }
    }
    result
}

/// SIMD-optimized matrix addition for ARM (NEON)
#[cfg(target_arch = "aarch64")]
unsafe fn simd_add_matrix_arm(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n: usize = a.len();
    let mut result: Vec<Vec<f64>> = vec![vec![0.0; n]; n];

    for i in 0..n {
        for j in (0..n).step_by(2) {
            let va: float64x2_t = vld1q_f64(&a[i][j]);  // Load 2 elements from matrix a
            let vb: float64x2_t = vld1q_f64(&b[i][j]);  // Load 2 elements from matrix b
            let vsum: float64x2_t = vaddq_f64(va, vb);  // Perform addition
            vst1q_f64(&mut result[i][j], vsum);  // Store result
        }
    }
    result
}

/// SIMD-optimized matrix subtraction for x86_64 (AVX)
#[cfg(target_arch = "x86_64")]
unsafe fn simd_sub_matrix_x86(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = a.len();
    let mut result = vec![vec![0.0; n]; n];

    for i in 0..n {
        for j in (0..n).step_by(4) {  // Process 4 elements at a time
            let va = _mm256_loadu_pd(&a[i][j]);  // Load 4 elements from matrix a
            let vb = _mm256_loadu_pd(&b[i][j]);  // Load 4 elements from matrix b
            let vres = _mm256_sub_pd(va, vb);    // Perform subtraction: a[i][j] - b[i][j]
            _mm256_storeu_pd(&mut result[i][j], vres);  // Store the result
        }
    }
    result
}

/// SIMD-optimized matrix subtraction for ARM (NEON)
#[cfg(target_arch = "aarch64")]
unsafe fn simd_sub_matrix_arm(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n: usize = a.len();
    let mut result: Vec<Vec<f64>> = vec![vec![0.0; n]; n];

    for i in 0..n {
        for j in (0..n).step_by(2) {  // Process 2 elements at a time
            let va: float64x2_t = vld1q_f64(&a[i][j]);  // Load 2 elements from matrix a
            let vb: float64x2_t = vld1q_f64(&b[i][j]);  // Load 2 elements from matrix b
            let vres: float64x2_t = vsubq_f64(va, vb);  // Perform subtraction: a[i][j] - b[i][j]
            vst1q_f64(&mut result[i][j], vres);  // Store the result
        }
    }
    result
}

fn sub_matrix(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
   // Use architecture-specific SIMD implementation
   #[cfg(target_arch = "x86_64")]
   {
       unsafe { simd_sub_matrix_x86(&a, &b) }
   }

   #[cfg(target_arch = "aarch64")]
   {
       unsafe { simd_sub_matrix_arm(&a, &b) }
   }

   // Fallback for unsupported architectures
   #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
   {
       panic!("Error: Unsupported Architecture");
   } 
}

fn add_matrix(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    // Use architecture-specific SIMD implementation
    #[cfg(target_arch = "x86_64")]
    {
        unsafe { simd_add_matrix_x86(&a, &b) }
    }

    #[cfg(target_arch = "aarch64")]
    {
        unsafe { simd_add_matrix_arm(&a, &b) }
    }

    // Fallback for unsupported architectures
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        panic!("Error: Unsupported Architecture");
    }
}

/// Strassen's matrix multiplication with a threshold for switching to traditional multiplication
fn matrix_mult_strassen(a: &[Vec<f64>], b: &[Vec<f64>], threshold: usize) -> Vec<Vec<f64>> {
    let n: usize = a.len();
    let m: usize = b[0].len();
    let p: usize = b.len();

    // Base case: Use traditional multiplication for small matrices
    if n <= threshold || m <= threshold || p <= threshold {
        return traditional_mult(a, b);
    }

    // Split the matrices into four submatrices (padding handled)
    let (a11, a12, a21, a22) = split_matrix(a);
    let (b11, b12, b21, b22) = split_matrix(b);

    // Calculate the 7 Strassen products
    let p1: Vec<Vec<f64>> = matrix_mult_strassen(&add_matrix(&a11, &a22), &add_matrix(&b11, &b22), threshold);
    let p2: Vec<Vec<f64>> = matrix_mult_strassen(&add_matrix(&a21, &a22), &b11, threshold);
    let p3: Vec<Vec<f64>> = matrix_mult_strassen(&a11, &sub_matrix(&b12, &b22), threshold);
    let p4: Vec<Vec<f64>> = matrix_mult_strassen(&a22, &sub_matrix(&b21, &b11), threshold);
    let p5: Vec<Vec<f64>> = matrix_mult_strassen(&add_matrix(&a11, &a12), &b22, threshold);
    let p6: Vec<Vec<f64>> = matrix_mult_strassen(&sub_matrix(&a21, &a11), &add_matrix(&b11, &b12), threshold);
    let p7: Vec<Vec<f64>> = matrix_mult_strassen(&sub_matrix(&a12, &a22), &add_matrix(&b21, &b22), threshold);

    // Compute the values of the resulting submatrices
    let c11: Vec<Vec<f64>> = add_matrix(&sub_matrix(&add_matrix(&p1, &p4), &p5), &p7);
    let c12: Vec<Vec<f64>> = add_matrix(&p3, &p5);
    let c21: Vec<Vec<f64>> = add_matrix(&p2, &p4);
    let c22: Vec<Vec<f64>> = add_matrix(&sub_matrix(&add_matrix(&p1, &p3), &p2), &p6);

    // Merge the resulting submatrices into the final matrix
    merge_matrix(c11, c12, c21, c22)
}

/// Strassen's matrix multiplication with padding for non-square matrices
fn strassen_with_padding(a: &[Vec<f64>], b: &[Vec<f64>], threshold: usize) -> Vec<Vec<f64>> {
    let a_rows: usize = a.len();
    let a_cols: usize = a[0].len();
    let b_rows: usize = b.len();
    let b_cols: usize = b[0].len();

    // Determine the size needed (nearest power of two based on the largest dimensions)
    let new_rows: usize = next_power_of_two(a_rows.max(b_rows));
    let new_cols: usize = next_power_of_two(a_cols.max(b_cols));

    // Pad matrices if necessary
    let a_padded: Vec<Vec<f64>> = pad_matrix(a, new_rows, new_cols);
    let b_padded: Vec<Vec<f64>> = pad_matrix(b, new_rows, new_cols);

    // Perform Strassen's multiplication
    let result_padded: Vec<Vec<f64>> = matrix_mult_strassen(&a_padded, &b_padded, threshold);

    // Unpad the result to get back to the original matrix size
    unpad_matrix(result_padded, a_rows, b_cols)
}

/// Merges four submatrices into a single matrix
fn merge_matrix(
    c11: Vec<Vec<f64>>, 
    c12: Vec<Vec<f64>>, 
    c21: Vec<Vec<f64>>, 
    c22: Vec<Vec<f64>>
) -> Vec<Vec<f64>> {
    let half_size: usize = c11.len();  // Size of submatrices
    let n: usize = half_size * 2;      // Size of the full matrix

    // Initialize the resulting matrix with the correct size
    let mut result: Vec<Vec<f64>> = vec![vec![0.0; n]; n];

    // Fill in the top-left (c11)
    for i in 0..half_size {
        for j in 0..half_size {
            result[i][j] = c11[i][j];
        }
    }

    // Fill in the top-right (c12)
    for i in 0..half_size {
        for j in 0..half_size {
            result[i][j + half_size] = c12[i][j];
        }
    }

    // Fill in the bottom-left (c21)
    for i in 0..half_size {
        for j in 0..half_size {
            result[i + half_size][j] = c21[i][j];
        }
    }

    // Fill in the bottom-right (c22)
    for i in 0..half_size {
        for j in 0..half_size {
            result[i + half_size][j + half_size] = c22[i][j];
        }
    }

    result
}

#[pyfunction]
fn sin(x: f64) -> PyResult<f64> {
    let result: f64 = chebyshev_sin(x);
    Ok(result)
}

#[pyfunction]
fn matrix_mult(a: Vec<Vec<f64>>, b: Vec<Vec<f64>>) -> PyResult<Vec<Vec<f64>>> {

    const THRESHOLD: usize = 2;

    // Perform Strassen's multiplication
    let result: Vec<Vec<f64>> = strassen_with_padding(&a, &b, THRESHOLD);
    Ok(result)
}

#[pyfunction]
fn sqrt(x: f64) -> f64 {
    newton_sqrt(x)
}

#[pyfunction]
fn dot_product(a: Vec<f32>, b: Vec<f32>) -> PyResult<f32> {
    let ans: f32 = dot_product_fe(&a, &b);
    Ok(ans)
    
}

#[pymodule]
fn ferrox(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(matrix_mult, m)?)?;
    m.add_function(wrap_pyfunction!(dot_product, m)?)?;
    Ok(())
}