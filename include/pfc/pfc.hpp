#pragma once

#include <chrono>
#include <climits>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <heffte.h>

namespace pfc {
namespace constants {

const double pi = std::atan(1.0) * 4.0;

// 'lattice' constants for the three ordered phases that exist in 2D/3D PFC
const double a1D = 2 * pi;               // stripes
const double a2D = 2 * pi * 2 / sqrt(3); // triangular
const double a3D = 2 * pi * sqrt(2);     // BCC

} // namespace constants
} // namespace pfc

namespace PFC {

template <typename T> size_t sizeof_vec(std::vector<T> &V) {
  return V.size() * sizeof(T);
}

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
  double saveat_ = 1.0;
  int n0 = 0;
  int max_iters = INT_MAX;
  std::string status_msg = "Initializing";
  size_t mem_allocated = 0;
  std::filesystem::path results_dir = ".";

  // use additional workspace to make calculation faster, increase memory usage
  const bool heffte_use_workspace = true;
  // Pointer to HeFFTe FFT
  heffte::fft3d_r2c<heffte::backend::fftw> *fft;

  // Temporary workspace to make FFT faster
  std::vector<std::complex<double>> wrk;

  // Data type used for writing subarray
  MPI_Datatype filetype;

  // This array is used to measure time during stepping
  std::array<double, 8> timing;

  void set_size(int Lx, int Ly, int Lz) {
    this->Lx = Lx;
    this->Ly = Ly;
    this->Lz = Lz;
  }

  void set_origin(double x0, double y0, double z0) {
    this->x0 = x0;
    this->y0 = y0;
    this->z0 = z0;
  }

  void set_dxdydz(double dx, double dy, double dz) {
    this->dx = dx;
    this->dy = dy;
    this->dz = dz;
  }

  void set_time(double t0, double t1, double dt) {
    this->t0 = t0;
    this->t1 = t1;
    this->dt = dt;
  }

  void set_saveat(const double saveat) { saveat_ = saveat; }
  double get_saveat() const { return saveat_; }

  virtual ~Simulation() {}

  virtual double get_dt(int, double) { return dt; }

  void set_max_iters(unsigned long int nmax) { max_iters = nmax; }

  void set_results_dir(std::string path) { results_dir = path; }

  std::filesystem::path get_results_dir() { return results_dir; }

  void set_fft(heffte::fft3d_r2c<heffte::backend::fftw> &fft_) { fft = &fft_; }

  void fft_r2c(std::vector<double> &A, std::vector<std::complex<double>> &B) {
    if (heffte_use_workspace) {
      fft->forward(A.data(), B.data(), wrk.data());
    } else {
      fft->forward(A.data(), B.data());
    }
  }

  void fft_c2r(std::vector<std::complex<double>> &A, std::vector<double> &B) {
    if (heffte_use_workspace) {
      fft->backward(A.data(), B.data(), wrk.data(), heffte::scale::full);
    } else {
      fft->backward(A.data(), B.data(), heffte::scale::full);
    }
  }

  void MPI_Read_Data(std::string filename, std::vector<double> &u) {
    MPI_File fh;
    MPI_File_open(MPI_COMM_WORLD, filename.c_str(), MPI_MODE_RDONLY,
                  MPI_INFO_NULL, &fh);
    MPI_File_set_view(fh, 0, MPI_DOUBLE, filetype, "native", MPI_INFO_NULL);
    MPI_File_read_all(fh, u.data(), u.size(), MPI_DOUBLE, MPI_STATUS_IGNORE);
    MPI_File_close(&fh);
  }

  void MPI_Write_Data(std::string filename, std::vector<double> &u) {
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

  virtual bool done(int n, double t) {
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

  virtual void set_step(int n) { n0 = n; }

  // in practice we are interested of replacing the things below with our
  // owns...

  virtual void allocate(size_t size_inbox, size_t size_outbox) = 0;

  virtual void prepare_operators(std::array<int, 3> low,
                                 std::array<int, 3> high) = 0;

  virtual void prepare_initial_condition(std::array<int, 3> low,
                                         std::array<int, 3> high) = 0;

  virtual void apply_bc(std::array<int, 3>, std::array<int, 3>){};

  virtual void step(int n, double t) = 0;

  virtual bool writeat(int, double) { return false; };

  virtual void write_results(int, double) { return; };
};

void MPI_Solve(Simulation &s) {

  std::cout << std::fixed;
  std::cout.precision(3);

  // unpack simulation settings
  const int Lx = s.Lx;
  const int Ly = s.Ly;
  const int Lz = s.Lz;

  MPI_Comm comm = MPI_COMM_WORLD;

  int me; // this process rank within the comm
  MPI_Comm_rank(comm, &me);
  s.me = me;

  int num_ranks; // total number of ranks in the comm
  MPI_Comm_size(comm, &num_ranks);
  s.num_ranks = num_ranks;

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

  s.set_fft(fft);

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

  s.filetype = filetype;

  // *** Allocate memory for workers. ***

  MPI_Barrier(MPI_COMM_WORLD);
  if (me == 0) {
    std::cout << std::endl
              << "***** MEMORY ALLOCATION STATUS *****" << std::endl;
  }
  s.allocate(fft.size_inbox(), fft.size_outbox());
  if (s.heffte_use_workspace) {
    // internal workspace used by HeFFTe to make FFT faster
    s.wrk.resize(fft.size_workspace());
  }
  const size_t mem_allocated = s.mem_allocated;
  const size_t mem_allocated_wrk = sizeof_vec(s.wrk);
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
    const long int ndofs = s.Lx * s.Ly * s.Lz;
    const double d = 1.0 / ndofs;
    const double size_perdof = d * total_size;
    const double size_wrk_perdof = d * total_size_wrk;
    const double size_total_perdof = d * (total_size + total_size_wrk);
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
    s.prepare_operators(outbox.low, outbox.high);
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
    s.prepare_initial_condition(inbox.low, inbox.high);
    s.apply_bc(inbox.low, inbox.high);
    MPI_Barrier(MPI_COMM_WORLD);
    t_init += MPI_Wtime();
    if (me == 0) {
      std::cout << "done in " << t_init << " seconds" << std::endl;
    }
  }
  int n = s.n0;
  double t = s.t0;

  double Sw = 0.0;
  if (n == 0) { // write initial condition
    Sw = -MPI_Wtime();
    MPI_Barrier(MPI_COMM_WORLD);
    s.write_results(n, t);
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
  const double alpha = 0.05;
  double S = 0.0;
  std::array<double, 8> timing;
  int t_saveat_cnt = 1;
  double t_saveat = s.t0 + t_saveat_cnt * s.get_saveat();

  while (!s.done(n, t)) {
    n += 1;
    t += s.get_dt(n, t);
    bool write_results = false;
    if (t >= t_saveat) {
      t = t_saveat;
      write_results = true;
    }
    if (me == 0) {
      std::cout << "***** STARTING STEP # " << n << " ***** at time " << t
                << std::endl;
    }

    double dt_step = -MPI_Wtime();
    MPI_Barrier(MPI_COMM_WORLD);
    s.apply_bc(inbox.low, inbox.high);
    MPI_Barrier(MPI_COMM_WORLD);
    s.step(n, t);
    MPI_Send(&(s.timing), 8, MPI_DOUBLE, 0, MPI_TAG_TIMING, MPI_COMM_WORLD);
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
#ifdef PFC_SHOW_TIMING_PER_WORKER
        std::cout << "MPI Worker " << i << ": FFT " << timing[1] << ", Other "
                  << timing[2] << ", Total " << timing[0] << std::endl;
#endif
        total_time += timing[0];
        fft_time += timing[1];
        other_time += timing[2];
      }
      total_time /= num_ranks;
      fft_time /= num_ranks;
      other_time /= num_ranks;
      std::cout << "Average time: FFT " << fft_time << ", Other " << other_time
                << ", Total " << total_time << std::endl;

      S = (n < 5) ? dt_step : alpha * dt_step + (1.0 - alpha) * S;
      auto n_left = (s.t1 - t) / s.get_dt(n, t);
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

    if (write_results) {
      double dt_write = -MPI_Wtime();
      MPI_Barrier(MPI_COMM_WORLD);
      s.write_results(t_saveat_cnt, t);
      MPI_Barrier(MPI_COMM_WORLD);
      dt_write += MPI_Wtime();
      if (me == 0) {
        Sw = alpha * dt_write + (1.0 - alpha) * Sw;
        std::cout << "Results writing time: " << dt_write
                  << " seconds (avg: " << Sw << " seconds)" << std::endl;
      }
      t_saveat_cnt += 1;
      t_saveat = s.t0 + t_saveat_cnt * s.get_saveat();
    }

    if (me == 0) {
      std::cout << "***** FINISHING STEP # " << n << " *****" << std::endl
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
    std::cout << "Simulation done. Status message: " + s.status_msg
              << std::endl;
  }
}

} // namespace PFC
