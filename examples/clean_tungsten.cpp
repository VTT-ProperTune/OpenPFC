// heFFTe implementation of pfc code

#include <chrono>
#include <climits>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <heffte.h>

template <typename T> size_t sizeof_vec(std::vector<T> &V) {
  return V.size() * sizeof(T);
}

typedef unsigned long index;

void MPI_Write_Data(std::string filename, MPI_Datatype &filetype,
                    std::vector<double> &u) {
  MPI_File fh;
  MPI_File_open(MPI_COMM_WORLD, filename.c_str(),
                MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);
  MPI_Offset filesize = 0;
  const unsigned int disp = 0;
  MPI_File_set_size(fh, filesize); // force overwriting existing data
  MPI_File_set_view(fh, disp, MPI_DOUBLE, filetype, "native", MPI_INFO_NULL);
  MPI_File_write_all(fh, u.data(), u.size(), MPI_DOUBLE, MPI_STATUS_IGNORE);
  MPI_File_close(&fh);
}

const double pi = std::atan(1.0) * 4.0;

// 'lattice' constants for the three ordered phases that exist in 2D/3D PFC
const double a1D = 2 * pi;               // stripes
const double a2D = 2 * pi * 2 / sqrt(3); // triangular
const double a3D = 2 * pi * sqrt(2);     // BCC

#define MPI_TAG_TIMING 3
#define MPI_TAG_INBOX_LOW 4
#define MPI_TAG_INBOX_HIGH 5
#define MPI_TAG_OUTBOX_LOW 6
#define MPI_TAG_OUTBOX_HIGH 7

struct Simulation {

  int me = 0;
  int num_ranks = 1;
  int Lx = 128;
  int Ly = 128;
  int Lz = 128;
  double dx = 1.0;
  double dy = 1.0;
  double dz = 1.0;
  double x0 = -64.0;
  double y0 = -64.0;
  double z0 = -64.0;
  double t0 = 0.0;
  double t1 = 10.0;
  double dt = 1.0;
  size_t max_iters = ULONG_MAX;
  std::string status_msg = "Initializing";
  size_t mem_allocated = 0;
  std::filesystem::path results_dir = ".";

  // Pointer to HeFFTe FFT
  heffte::fft3d_r2c<heffte::backend::fftw> *fft;
  // Temporary workspace to make FFT faster
  std::vector<std::complex<double>> wrk;

  // This array is used to measure time during stepping
  std::array<double, 8> timing;

  void set_size(int Lx_, int Ly_, int Lz_) {
    Lx = Lx_;
    Ly = Ly_;
    Lz = Lz_;
  }

  void set_origin(double x0_, double y0_, double z0_) {
    x0 = x0_;
    y0 = y0_;
    z0 = z0_;
  }

  void set_dxdydz(double dx_, double dy_, double dz_) {
    dx = dx_;
    dy = dy_;
    dz = dz_;
  }

  void set_time(double t0_, double t1_, double dt_) {
    t0 = t0_;
    t1 = t1_;
    dt = dt_;
  }

  virtual double get_dt(size_t n, double t) {
    return dt;
  }

  void set_max_iters(unsigned long int nmax) {
    max_iters = nmax;
  }

  void set_results_dir(std::string path) {
    results_dir = path;
  }

  std::filesystem::path get_results_dir() {
    return results_dir;
  }

  void set_fft(heffte::fft3d_r2c<heffte::backend::fftw> &fft_) {
    fft = &fft_;
  }

  void fft_r2c(std::vector<double> &A, std::vector<std::complex<double>> &B) {
    fft->forward(A.data(), B.data(), wrk.data());
  }

  void fft_c2r(std::vector<std::complex<double>> &A, std::vector<double> &B) {
    fft->backward(A.data(), B.data(), wrk.data(), heffte::scale::full);
  }

  virtual bool done(unsigned long int n, double t) {
    if (n >= max_iters) {
      status_msg = "maximum number of iterations (" +
                   std::to_string(max_iters) + ") reached";
      return true;
    }
    if (t >= t1) {
      status_msg = "simulated succesfully to time " + std::to_string(t1) +
                   ", (" + std::to_string(n) + " iterations)";
      return true;
    }
    return false;
  }

  // in practice we are interested of replacing the things below with our
  // owns...

  virtual void allocate(size_t size_inbox, size_t size_outbox) = 0;

  virtual void prepare_operators(std::array<int, 3> low,
                                 std::array<int, 3> high) = 0;

  virtual void prepare_initial_condition(std::array<int, 3> low,
                                         std::array<int, 3> high) = 0;

  virtual void step(size_t n, double t) = 0;

  virtual bool writeat(size_t n, double t) = 0;

  virtual void write_results(size_t n, double t, MPI_Datatype &filetype) = 0;
};

struct Tungsten : Simulation {

  // average density of the metastable fluid
  double n0 = -0.4;

  // Bulk densities at coexistence, obtained from phase diagram for chosen
  // temperature
  double n_sol = -0.047;
  double n_vap = -0.464;

  // Effective temperature parameters. Temperature in K. Remember to change
  // n_sol and n_vap according to phase diagram when T is changed.
  double T = 3300.0;
  double T0 = 156000.0;
  double Bx = 0.8582;

  // parameters that affect elastic and interface energies

  // width of C2's peak
  double alpha = 0.50;

  // how much we allow the k=1 peak to affect the k=0 value of the
  // correlation, by changing the higher order components of the Gaussian
  // function
  double alpha_farTol = 1.0 / 1000.0;

  // power of the higher order component of the gaussian function. Should be a
  // multiple of 2. Setting this to zero also disables the tolerance setting.
  int alpha_highOrd = 4;

  // derived dimensionless values used in calculating vapor model parameters
  double tau = T / T0;

  // Strength of the meanfield filter. Avoid values higher than ~0.28, to
  // avoid lattice-wavelength variations in the mean field
  double lambda = 0.22;

  // numerical stability parameter for the exponential integrator method
  double stabP = 0.2;

  // Vapor-model parameters
  double shift_u = 0.3341;
  double shift_s = 0.1898;

  double p2 = 1.0;
  double p3 = -1.0 / 2.0;
  double p4 = 1.0 / 3.0;
  double p2_bar = p2 + 2 * shift_s * p3 + 3 * pow(shift_s, 2) * p4;
  double p3_bar = shift_u * (p3 + 3 * shift_s * p4);
  double p4_bar = pow(shift_u, 2) * p4;

  double q20 = -0.0037;
  double q21 = 1.0;
  double q30 = -12.4567;
  double q31 = 20.0;
  double q40 = 45.0;

  double q20_bar = q20 + 2.0 * shift_s * q30 + 3.0 * pow(shift_s, 2) * q40;
  double q21_bar = q21 + 2.0 * shift_s * q31;
  double q30_bar = shift_u * (q30 + 3.0 * shift_s * q40);
  double q31_bar = shift_u * q31;
  double q40_bar = pow(shift_u, 2) * q40;

  double q2_bar = q21_bar * tau + q20_bar;
  double q3_bar = q31_bar * tau + q30_bar;
  double q4_bar = q40_bar;

  // we will allocate these arrays later on
  std::vector<double> filterMF, opL, opN;
  std::vector<double> psiMF, psi, psiN;
  std::vector<std::complex<double>> psiMF_F, psi_F, psiN_F;

  // used to measure execution time of step and write_results
  std::array<double, 10> timers;

  /*
    This function is ran only one time during the initialization of solver. Used
    to allocate all necessary arrays.
   */
  void allocate(size_t size_inbox, size_t size_outbox) {

    // operators are only half size due to the symmetry of fourier space
    filterMF.resize(size_outbox);
    opL.resize(size_outbox);
    opN.resize(size_outbox);

    // psi, psiMF, psiN, where suffix F means in fourier space
    psi.resize(size_inbox);
    psi_F.resize(size_outbox);
    psiMF.resize(size_inbox);
    psiMF_F.resize(size_outbox);
    psiN.resize(size_inbox);
    psiN_F.resize(size_outbox);

    // At the end, let's calculate how much did we allocate memory
    mem_allocated = sizeof_vec(filterMF) + sizeof_vec(opL) + sizeof_vec(opN) +
                    sizeof_vec(psi) + sizeof_vec(psiMF) + sizeof_vec(psiN) +
                    sizeof_vec(psi_F) + sizeof_vec(psiMF_F) +
                    sizeof_vec(psiN_F);

    // should be equal to
    // size_inbox*8*3     (psi, psiMF, psiN)
    // size_outbox*16*3   (psi_F, psiMF_F, psiN_F)
    // size_outbox*8*3    (filterMF, opL, opN)
  }

  /*
    This function is called after allocate(), used to fill operators.
  */
  void prepare_operators(std::array<int, 3> low, std::array<int, 3> high) {

    // prepare the linear and non-linear operators

    index idx = 0;
    const double fx = 2.0 * pi / (dx * Lx);
    const double fy = 2.0 * pi / (dy * Ly);
    const double fz = 2.0 * pi / (dz * Lz);

    for (index k = low[2]; k <= high[2]; k++) {
      for (index j = low[1]; j <= high[1]; j++) {
        for (index i = low[0]; i <= high[0]; i++) {

          // laplacian operator -k^2
          const double ki = (i <= Lx / 2) ? i * fx : (i - Lx) * fx;
          const double kj = (j <= Ly / 2) ? j * fy : (j - Ly) * fy;
          const double kk = (k <= Lz / 2) ? k * fz : (k - Lz) * fz;
          const double kLap = -(ki * ki + kj * kj + kk * kk);

          // mean-field filtering operator (chi) make a C2 that's quasi-gaussian
          // on the left, and ken-style on the right
          const double alpha2 = 2.0 * alpha * alpha;
          const double lambda2 = 2.0 * lambda * lambda;
          const double fMF = exp(kLap / lambda2);
          const double k = sqrt(-kLap) - 1.0;
          const double k2 = k * k;

          double g1 = 0;
          if (alpha_highOrd == 0) {
            // gaussian peak
            g1 = exp(-k2 / alpha2);
          } else {
            // quasi-gaussian peak with higher order component
            // to make it decay faster towards k=0
            double rTol = -alpha2 * log(alpha_farTol) - 1.0;
            g1 = exp(-(k2 + rTol * pow(k, alpha_highOrd)) / alpha2);
          }
          // taylor expansion of gaussian peak to order 2
          const double g2 = 1.0 - 1.0 / alpha2 * k2;
          // splice the two sides of the peak
          const double gf = (k < 0.0) ? g1 : g2;

          // we separate this out because it is needed in the nonlinear
          // calculation when T is not constant in space
          const double opPeak = -Bx * exp(-T / T0) * gf;

          // includes the lowest order n_mf term since it is a linear term
          const double opCk = stabP + p2_bar + opPeak + q2_bar * fMF;

          filterMF[idx] = fMF;
          opL[idx] = exp(kLap * opCk * dt);
          opN[idx] = (opCk == 0.0) ? kLap * dt : (opL[idx] - 1.0) / opCk;

          idx += 1;
        }
      }
    }
  }

  /*
    PFC stepping function

    Timing:
    timing[0]   total time used in step
    timing[1]   time used in FFT
    timing[2]   time used in all other things
  */
  void step(size_t n, double t) {

    for (index i = 0; i < 8; i++) {
      timing[i] = 0.0;
    }

    timing[0] -= MPI_Wtime();

    // Calculate mean-field density n_mf
    timing[2] -= MPI_Wtime();
    for (size_t idx = 0; idx < psiMF_F.size(); idx++) {
      psiMF_F[idx] = filterMF[idx] * psi_F[idx];
    }
    timing[2] += MPI_Wtime();

    timing[1] -= MPI_Wtime();
    fft_c2r(psiMF_F, psiMF);
    timing[1] += MPI_Wtime();

    // Calculate the nonlinear part of the evolution equation in a real space
    timing[2] -= MPI_Wtime();
    for (size_t idx = 0; idx < psiN.size(); idx++) {
      const double u = psi[idx];
      const double v = psiMF[idx];
      const double u2 = u * u;
      const double u3 = u2 * u;
      const double v2 = v * v;
      const double v3 = v2 * v;
      psiN[idx] = p3_bar * u2 + p4_bar * u3 + q3_bar * v2 + q4_bar * v3;
    }
    timing[2] += MPI_Wtime();

    // Apply stabilization factor if given in parameters
    timing[2] -= MPI_Wtime();
    if (stabP != 0.0) {
      for (index idx = 0; idx < psiN.size(); idx++) {
        psiN[idx] = psiN[idx] - stabP * psi[idx];
      }
    }
    timing[2] += MPI_Wtime();

    // Fourier transform of the nonlinear part of the evolution equation
    timing[1] -= MPI_Wtime();
    fft_r2c(psiN, psiN_F);
    timing[1] += MPI_Wtime();

    // Apply one step of the evolution equation
    timing[2] -= MPI_Wtime();
    for (size_t idx = 0; idx < psi_F.size(); idx++) {
      psi_F[idx] = opL[idx] * psi_F[idx] + opN[idx] * psiN_F[idx];
    }
    timing[2] += MPI_Wtime();

    // Inverse Fourier transform result back to real space
    timing[1] -= MPI_Wtime();
    fft_c2r(psi_F, psi);
    timing[1] += MPI_Wtime();

    timing[0] += MPI_Wtime();
  }

  /*
  Initial condition is defined here
  */
  void prepare_initial_condition(std::array<int, 3> low,
                                 std::array<int, 3> high) {
    // calculating approx amplitude. This is related to the phase diagram
    // calculations.
    const double rho_seed = n_sol;
    const double A_phi = 135.0 * p4_bar;
    const double B_phi = 16.0 * p3_bar + 48.0 * p4_bar * rho_seed;
    const double C_phi = -6.0 * (Bx * exp(-T / T0)) + 6.0 * p2_bar +
                         12.0 * p3_bar * rho_seed +
                         18.0 * p4_bar * pow(rho_seed, 2);
    const double d = abs(9.0 * pow(B_phi, 2) - 32.0 * A_phi * C_phi);
    const double amp_eq = (-3.0 * B_phi + sqrt(d)) / (8.0 * A_phi);

    const double s = 1.0 / sqrt(2.0);
    const std::array<double, 3> q1 = {s, s, 0};
    const std::array<double, 3> q2 = {s, 0, s};
    const std::array<double, 3> q3 = {0, s, s};
    const std::array<double, 3> q4 = {s, 0, -s};
    const std::array<double, 3> q5 = {s, -s, 0};
    const std::array<double, 3> q6 = {0, s, -s};
    const std::array<std::array<double, 3>, 6> q = {q1, q2, q3, q4, q5, q6};

    index idx = 0;
    // double r2 = pow(0.2 * (Lx * dx), 2);
    const double r2 = pow(64.0, 2);
    double u;
    for (index k = low[2]; k <= high[2]; k++) {
      for (index j = low[1]; j <= high[1]; j++) {
        for (index i = low[0]; i <= high[0]; i++) {
          const double x = x0 + i * dx;
          const double y = y0 + j * dy;
          const double z = z0 + k * dz;
          const bool seedmask = x * x + y * y + z * z < r2;
          if (!seedmask) {
            u = n0;
          } else {
            u = rho_seed;
            for (int i = 0; i < 6; i++) {
              u += 2.0 * amp_eq * cos(q[i][0] * x + q[i][1] * y + q[i][2] * z);
            }
          }
          psi[idx] = u;
          idx += 1;
        }
      }
    }

    // Calculate FFT for initial field, psi_F = fft(psi)
    fft_r2c(psi, psi_F);
  }

  bool writeat(size_t n, double t) {
    return n % 10000 == 0 | t >= t1;
  }

  /*
 Results writing routine
 */
  void write_results(size_t n, double t, MPI_Datatype &filetype) {
    auto filename = results_dir / ("u" + std::to_string(n) + ".bin");
    if (me == 0) {
      std::cout << "Writing results to " << filename << std::endl;
    }
    // Apply inverse fourier transform to density field, psi_F = fft^-1(psi)
    fft_c2r(psi_F, psi);
    MPI_Write_Data(filename, filetype, psi);
  };

}; // end of class

void MPI_Solve(Simulation *s) {

  std::cout << std::fixed;
  std::cout.precision(3);

  // unpack simulation settings
  const int Lx = s->Lx;
  const int Ly = s->Ly;
  const int Lz = s->Lz;

  MPI_Comm comm = MPI_COMM_WORLD;

  int me; // this process rank within the comm
  MPI_Comm_rank(comm, &me);
  s->me = me;

  int num_ranks; // total number of ranks in the comm
  MPI_Comm_size(comm, &num_ranks);
  s->num_ranks = num_ranks;

  if (me == 0) {
    std::cout << "***** PFC SIMULATOR USING HEFFTE *****" << std::endl
              << std::endl;
  }

  /*
  If the input of an FFT transform consists of all real numbers,
   the output comes in conjugate pairs which can be exploited to reduce
   both the floating point operations and MPI communications.
   Given a global set of indexes, HeFFTe can compute the corresponding DFT
   and exploit the real-to-complex symmetry by selecting a dimension
   and reducing the indexes by roughly half (the exact formula is floor(n / 2)
  + 1).
   */
  const int Lx_c = floor(Lx / 2) + 1;
  // the dimension where the data will shrink
  const int r2c_direction = 0;
  // define real doman
  heffte::box3d<> real_indexes({0, 0, 0}, {Lx - 1, Ly - 1, Lz - 1});
  // define complex domain
  heffte::box3d<> complex_indexes({0, 0, 0}, {Lx_c - 1, Ly - 1, Lz - 1});

  // check if the complex indexes have correct dimension
  assert(real_indexes.r2c(r2c_direction) == complex_indexes);

  // create a processor grid with minimum surface (measured in number of
  // indexes)
  auto proc_grid = heffte::proc_setup_min_surface(real_indexes, num_ranks);

  // split all indexes across the processor grid, defines a set of boxes
  auto real_boxes = heffte::split_world(real_indexes, proc_grid);
  auto complex_boxes = heffte::split_world(complex_indexes, proc_grid);

  // pick the box corresponding to this rank
  heffte::box3d<> const inbox = real_boxes[me];
  heffte::box3d<> const outbox = complex_boxes[me];

  // define the heffte class and the input and output geometry
  heffte::fft3d_r2c<heffte::backend::fftw> fft(inbox, outbox, r2c_direction,
                                               comm);

  s->set_fft(fft);

  // *** Report domain decomposition status ***
  MPI_Send(&(inbox.low), 3, MPI_INT, 0, MPI_TAG_INBOX_LOW, MPI_COMM_WORLD);
  MPI_Send(&(inbox.high), 3, MPI_INT, 0, MPI_TAG_INBOX_HIGH, MPI_COMM_WORLD);
  MPI_Send(&(outbox.low), 3, MPI_INT, 0, MPI_TAG_OUTBOX_LOW, MPI_COMM_WORLD);
  MPI_Send(&(outbox.high), 3, MPI_INT, 0, MPI_TAG_OUTBOX_HIGH, MPI_COMM_WORLD);
  if (me == 0) {
    std::cout << "***** DOMAIN DECOMPOSITION STATUS *****" << std::endl;
    std::cout << "Contructed Fourier transform from REAL to COMPLEX, using "
                 "real-to-complex symmetry."
              << std::endl;

    std::cout << "Grid in real space: [" << Lx << ", " << Ly << ", " << Lz
              << "] (" << real_indexes.count() << " indexes)" << std::endl;
    std::cout << "Grid in complex space: [" << Lx_c << ", " << Ly << ", " << Lz
              << "] (" << complex_indexes.count() << " indexes)" << std::endl;
    std::cout << "Domain is split into " << num_ranks
              << " parts, with minimum surface processor grid: ["
              << proc_grid[0] << ", " << proc_grid[1] << ", " << proc_grid[2]
              << "]" << std::endl;
    std::array<int, 3> in_low, in_high, out_low, out_high;
    for (int i = 0; i < num_ranks; i++) {
      MPI_Recv(&in_low, 3, MPI_INT, i, MPI_TAG_INBOX_LOW, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
      MPI_Recv(&in_high, 3, MPI_INT, i, MPI_TAG_INBOX_HIGH, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
      MPI_Recv(&out_low, 3, MPI_INT, i, MPI_TAG_OUTBOX_LOW, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
      MPI_Recv(&out_high, 3, MPI_INT, i, MPI_TAG_OUTBOX_HIGH, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
      size_t size_in = (in_high[0] - in_low[0] + 1) *
                       (in_high[1] - in_low[1] + 1) *
                       (in_high[2] - in_low[2] + 1);
      size_t size_out = (out_high[0] - out_low[0] + 1) *
                        (out_high[1] - out_low[1] + 1) *
                        (out_high[2] - out_low[2] + 1);
      std::cout << "MPI Worker " << i << ", [" << in_low[0] << ", "
                << in_high[0] << "] x [" << in_low[1] << ", " << in_high[1]
                << "] x [" << in_low[2] << ", " << in_high[2] << "] ("
                << size_in << " indexes) => [" << out_low[0] << ", "
                << out_high[0] << "] x [" << out_low[1] << ", " << out_high[1]
                << "] x [" << out_low[2] << ", " << out_high[2] << "] ("
                << size_out << " indexes)" << std::endl;
    }
  }

  // *** Create and commit new data type ***

  MPI_Datatype filetype;
  const int size_array[] = {Lx, Ly, Lz};
  const int subsize_array[] = {inbox.high[0] - inbox.low[0] + 1,
                               inbox.high[1] - inbox.low[1] + 1,
                               inbox.high[2] - inbox.low[2] + 1};
  const int start_array[] = {inbox.low[0], inbox.low[1], inbox.low[2]};
  MPI_Type_create_subarray(3, size_array, subsize_array, start_array,
                           MPI_ORDER_FORTRAN, MPI_DOUBLE, &filetype);
  MPI_Type_commit(&filetype);

  // *** Allocate memory for workers. ***

  MPI_Barrier(MPI_COMM_WORLD);
  if (me == 0) {
    std::cout << std::endl
              << "***** MEMORY ALLOCATION STATUS *****" << std::endl;
  }
  s->allocate(fft.size_inbox(), fft.size_outbox());
  // internal workspace used by HeFFTe to make FFT faster
  s->wrk.resize(fft.size_workspace());
  size_t mem_allocated = s->mem_allocated;
  size_t mem_allocated_wrk = sizeof_vec(s->wrk);
  MPI_Send(&mem_allocated, 1, MPI_LONG_INT, 0, 0, MPI_COMM_WORLD);
  MPI_Send(&mem_allocated_wrk, 1, MPI_LONG_INT, 0, 1, MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);
  if (me == 0) {
    size_t size = 0;
    size_t size_wrk = 0;
    size_t total_size = 0;
    size_t total_size_wrk = 0;
    for (int i = 0; i < num_ranks; i++) {
      MPI_Recv(&size, 1, MPI_LONG_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Recv(&size_wrk, 1, MPI_LONG_INT, i, 1, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
      std::cout << "MPI Worker " << i << ", " << (size + size_wrk)
                << " bytes allocated (" << size << " bytes user data, "
                << size_wrk << " bytes for fft workspace)" << std::endl;
      total_size += size;
      total_size_wrk += size_wrk;
    }
    int dim = s->Lx * s->Ly * s->Lz;
    double size_perdof = 1.0 * total_size / dim;
    double size_wrk_perdof = 1.0 * total_size_wrk / dim;
    double size_total_perdof = 1.0 * (total_size + total_size_wrk) / dim;
    std::cout << "Total " << total_size << " bytes allocated for user data ("
              << size_perdof << " bytes / dof)" << std::endl;
    std::cout << "Total " << total_size_wrk
              << " bytes allocated for workspace (" << size_wrk_perdof
              << " bytes / dof)" << std::endl;
    std::cout << "Total " << (total_size + total_size_wrk)
              << " bytes allocated (" << size_total_perdof << " bytes / dof)"
              << std::endl;
    std::cout << std::endl;
  }
  MPI_Barrier(MPI_COMM_WORLD);

  if (me == 0) {
    std::cout << "***** INITIALIZE SIMULATION *****" << std::endl;
  }

  {
    if (me == 0) {
      std::cout << "Preparing operators ... ";
    }
    MPI_Barrier(MPI_COMM_WORLD);
    double t_op = -MPI_Wtime();
    s->prepare_operators(outbox.low, outbox.high);
    MPI_Barrier(MPI_COMM_WORLD);
    t_op += MPI_Wtime();
    if (me == 0) {
      std::cout << "done in " << t_op << " seconds" << std::endl;
    }
  }

  {
    if (me == 0) {
      std::cout << "Generating initial condition ... ";
    }
    MPI_Barrier(MPI_COMM_WORLD);
    double t_init = -MPI_Wtime();
    s->prepare_initial_condition(inbox.low, inbox.high);
    MPI_Barrier(MPI_COMM_WORLD);
    t_init += MPI_Wtime();
    if (me == 0) {
      std::cout << "done in " << t_init << " seconds" << std::endl;
    }
  }
  index n = 0;
  double t = s->t0;

  double Sw = 0.0;
  if (s->writeat(n, t)) {
    Sw = -MPI_Wtime();
    MPI_Barrier(MPI_COMM_WORLD);
    s->write_results(n, t, filetype);
    MPI_Barrier(MPI_COMM_WORLD);
    Sw += MPI_Wtime();
    if (me == 0) {
      std::cout << "Results writing time: " << Sw << " seconds" << std::endl;
    }
  }

  if (me == 0) {
    std::cout << std::endl
              << "***** STARTING SIMULATION ***** " << std::endl
              << std::endl;
  }

  auto start = std::chrono::high_resolution_clock::now();

  // for timing step time
  const double alpha = 0.5;
  double S = 0.0;
  std::array<double, 8> timing;

  while (!s->done(n, t)) {
    n += 1;
    t += s->get_dt(n, t);
    if (me == 0) {
      std::cout << "***** STARTING STEP # " << n << " *****" << std::endl;
    }

    double dt_step = -MPI_Wtime();
    MPI_Barrier(MPI_COMM_WORLD);
    s->step(n, t);
    MPI_Send(&(s->timing), 8, MPI_DOUBLE, 0, MPI_TAG_TIMING, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    dt_step += MPI_Wtime();

    if (me == 0) {

      std::cout << "Step finished. Timing information:" << std::endl;
      double total_time = 0.0;
      double fft_time = 0.0;
      double other_time = 0.0;
      for (int i = 0; i < num_ranks; i++) {
        MPI_Recv(&timing, 8, MPI_DOUBLE, i, MPI_TAG_TIMING, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
        std::cout << "MPI Worker " << i << ": FFT " << timing[1] << ", Other "
                  << timing[2] << ", Total " << timing[0] << std::endl;
        total_time += timing[0];
        fft_time += timing[1];
        other_time += timing[2];
      }
      total_time /= num_ranks;
      fft_time /= num_ranks;
      other_time /= num_ranks;
      std::cout << "Average time: FFT " << fft_time << ", Other " << other_time
                << ", Total " << total_time << std::endl;

      S = (n == 1) ? dt_step : alpha * dt_step + (1.0 - alpha) * S;
      auto n_left = (s->t1 - t) / s->get_dt(n, t);
      auto eta = S * n_left;
      std::cout << "Step execution time: " << dt_step
                << " seconds. Average step execution time: " << S << " seconds."
                << std::endl;
      auto pct_done = 100.0 * n / (n + n_left);
      std::cout << "Simulation time: " << t << " seconds. " << n
                << " of estimated " << (n + n_left) << " steps (" << pct_done
                << " %) done." << std::endl;
      std::cout << "Simulation is estimated to be ready in " << eta
                << " seconds." << std::endl;
    }

    if (s->writeat(n, t)) {
      double dt_write = -MPI_Wtime();
      MPI_Barrier(MPI_COMM_WORLD);
      s->write_results(n, t, filetype);
      MPI_Barrier(MPI_COMM_WORLD);
      dt_write += MPI_Wtime();
      if (me == 0) {
        Sw = alpha * dt_write + (1.0 - alpha) * Sw;
        std::cout << "Results writing time: " << dt_write
                  << " seconds (avg: " << Sw << " seconds)" << std::endl;
      }
    }

    if (me == 0) {
      std::cout << "***** FINISING STEP # " << n << " *****" << std::endl
                << std::endl;
      std::cout.flush();
    }
  }
  auto stop = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
  if (me == 0) {
    std::cout << n << " iterations in " << duration.count() / 1000.0
              << " seconds (" << duration.count() / n << " ms / iteration)"
              << std::endl;
  }
  if (me == 0) {
    std::cout << "Simulation done. Status message: " + s->status_msg
              << std::endl;
  }
}

int main(int argc, char *argv[]) {

  // Let's define simulation settings, that are kind of standard for all types
  // of simulations. At least we need to define the world size and time.
  // Even spaced grid is used, thus we have something like x = x0 + dx*i for
  // spatial coordinate and t = t0 + dt*n for time.

  Simulation *s = new Tungsten();

  int Lx = 128;
  int Ly = 128;
  int Lz = 128;
  s->set_size(Lx, Ly, Lz);

  double dx = a3D / 8.0;
  double dy = a3D / 8.0;
  double dz = a3D / 8.0;
  s->set_dxdydz(dx, dy, dz);

  double x0 = -0.5 * Lx * dx;
  double y0 = -0.5 * Ly * dy;
  double z0 = -0.5 * Lz * dz;
  s->set_origin(x0, y0, z0);

  double t0 = 0.0;
  // double t1 = 200000.0;
  double t1 = 10.0;
  double dt = 1.0;
  s->set_time(t0, t1, dt);

  // define where to store results
  s->set_results_dir(
      "/home/juajukka/results/pfc-heffte-clean_tungsten_3d_1024x1024x1024");

  MPI_Init(&argc, &argv);
  MPI_Solve(s);
  MPI_Finalize();

  return 0;
}
