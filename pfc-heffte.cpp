// heFFTe implementation of pfc code

#include <argparse/argparse.hpp>
#include <heffte.h>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

double fRand(double fMin, double fMax) {
  double f = (double)rand() / RAND_MAX;
  return fMin + f * (fMax - fMin);
}

constexpr double pi() {
  return std::atan(1) * 4;
}

void compute_dft(MPI_Comm comm, json settings) {
  // unpack simulation and model settings
  const int nx = settings["nx"];
  const int ny = settings["ny"];
  const int nz = settings["nz"];
  const int N = settings["niters"];
  const double h = settings["h"];
  const double tau = settings["tau"];
  const double C = settings["C"];
  const double delta = settings["delta"];

  int me; // this process rank within the comm
  MPI_Comm_rank(comm, &me);

  int num_ranks; // total number of ranks in the comm
  MPI_Comm_size(comm, &num_ranks);

  if (me == 0) {
    std::cout << "Simulation settings:\n\n";
    std::cout << settings.dump(4) << "\n\n";
  }

  /*
  If the input of an FFT transform consists of all real numbers,
   the output comes in conjugate pairs which can be exploited to reduce
   both the floating point operations and MPI communications.
   Given a global set of indexes, HeFFTe can compute the corresponding DFT
   and exploit the real-to-complex symmetry by selecting a dimension
   and reducing the indexes by roughly half (the exact formula is floor(n / 2) +
  1).
   */
  const int nx_c = floor(nx / 2) + 1;
  // the dimension where the data will shrink
  const int r2c_direction = 0;
  // define real doman
  heffte::box3d<> real_indexes({0, 0, 0}, {nx - 1, ny - 1, nz - 1});
  // define complex domain
  heffte::box3d<> complex_indexes({0, 0, 0}, {nx_c - 1, ny - 1, nz - 1});

  // check if the complex indexes have correct dimension
  assert(real_indexes.r2c(r2c_direction) == complex_indexes);

  // report the indexes
  if (me == 0) {
    std::cout << "The global input contains " << real_indexes.count()
              << " real indexes.\n";
    std::cout << "The global output contains " << complex_indexes.count()
              << " complex indexes.\n";
  }

  // create a processor grid with minimum surface (measured in number of
  // indexes)
  auto proc_grid = heffte::proc_setup_min_surface(real_indexes, num_ranks);
  if (me == 0) {
    std::cout << "Minimum surface processor grid: [" << proc_grid[0] << ", "
              << proc_grid[1] << ", " << proc_grid[2] << "]" << std::endl;
  }

  // split all indexes across the processor grid, defines a set of boxes
  auto real_boxes = heffte::split_world(real_indexes, proc_grid);
  auto complex_boxes = heffte::split_world(complex_indexes, proc_grid);

  // pick the box corresponding to this rank
  heffte::box3d<> const inbox = real_boxes[me];
  heffte::box3d<> const outbox = complex_boxes[me];

  // define the heffte class and the input and output geometry
  heffte::fft3d_r2c<heffte::backend::fftw> fft(inbox, outbox, r2c_direction,
                                               comm);

  // vectors with the correct sizes to store the input and output data
  // taking the size of the input and output boxes
  std::cout << "Rank " << me << " input box: " << fft.size_inbox()
            << " indexes, indices x = [" << inbox.low[0] << ", "
            << inbox.high[0] << "], y = [" << inbox.low[1] << ", "
            << inbox.high[1] << "], "
            << "z = [" << inbox.low[2] << ", " << inbox.high[2]
            << "], outbox box: " << fft.size_outbox()
            << " indexes, indices x = [" << outbox.low[0] << ", "
            << outbox.high[0] << "], y = [" << outbox.low[1] << ", "
            << outbox.high[1] << "], "
            << "z = [" << outbox.low[2] << ", " << outbox.high[2] << "]"
            << std::endl;
  std::vector<double> U(fft.size_inbox());
  std::vector<std::complex<double>> C1(fft.size_outbox());
  std::vector<std::complex<double>> C2(fft.size_outbox());

  if (me == 0) {
    std::cout << "Generate input data" << std::endl;
  }
  for (size_t i = 0; i < U.size(); i++) {
    U[i] = fRand(-0.5, 0.5);
  }

  // set the strides for the triple indexes
  int local_plane = outbox.size[0] * outbox.size[1];
  int local_stride = outbox.size[0];

  // define workspace to improve performance
  std::vector<std::complex<double>> workspace(fft.size_workspace());

  if (me == 0) {
    std::cout << "Starting simulation\n\n";
  }

  {
    MPI_Datatype filetype;
    const int gdims[] = {nx, ny, nz};
    const auto lx = inbox.high[0] - inbox.low[0];
    const auto ly = inbox.high[1] - inbox.low[1];
    const auto lz = inbox.high[2] - inbox.low[2];
    const int ldims[] = {lx, ly, lz};
    const auto ox = inbox.low[0];
    const auto oy = inbox.low[1];
    const auto oz = inbox.low[2];
    const int offset[] = {ox, oy, oz};
    MPI_Type_create_subarray(3, gdims, ldims, offset, MPI_ORDER_C, MPI_DOUBLE,
                             &filetype);
    MPI_Type_commit(&filetype);
    const unsigned int disp = 0;

    MPI_File fh;
    const std::string filename = "0.bin";
    MPI_File_open(MPI_COMM_WORLD, filename.c_str(),
                  MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);
    MPI_Offset filesize = 0;
    MPI_File_set_size(fh, filesize); // force overwriting existing data
    MPI_File_set_view(fh, disp, MPI_DOUBLE, filetype, "native", MPI_INFO_NULL);
    MPI_File_write_all(fh, U.data(), U.size(), MPI_DOUBLE, MPI_STATUS_IGNORE);
    MPI_File_close(&fh);
  }

  // start iterations
  for (auto n = 1; n <= N; n++) {
    if (me == 0) {
      std::cout << "Iteration " << n << std::endl;
    }
    // fft.forward(U.data(), C1.data(), workspace.data(), heffte::scale::full);
    fft.forward(U.data(), C1.data(), workspace.data());
    for (auto i = 0; i < U.size(); i++) {
      U[i] = pow(U[i], 3.0) - C * U[i];
    };
    // fft.forward(U.data(), C2.data(), workspace.data(), heffte::scale::full);
    fft.forward(U.data(), C2.data(), workspace.data());

    // note the order of the loops corresponding to the default order (0, 1, 2)
    // order (0, 1, 2) means that the data in dimension 0 is contiguous
    for (auto i = outbox.low[2]; i <= outbox.high[2]; i++) {
      for (auto j = outbox.low[1]; j <= outbox.high[1]; j++) {
        for (auto k = outbox.low[0]; k <= outbox.high[0]; k++) {
          // triple indexes to linear index conversion
          auto idx = (i - outbox.low[2]) * local_plane +
                     (j - outbox.low[1]) * local_stride + k - outbox.low[0];
          auto c1 = cos(2 * pi() / nx * i);
          auto c2 = cos(2 * pi() / ny * j);
          auto c3 = cos(2 * pi() / nz * k);
          auto f = 3.0 / pow(h, 3.0) * (c1 + c2 + c3 - 3);
          auto d = 1.0 - tau * f * (pow(f + 1.0, 2.0) + C - delta);
          C1[idx] = (C1[idx] + tau * f * C2[i]) / d;
        }
      }
    }

    fft.backward(C1.data(), U.data(), workspace.data(), heffte::scale::full);
  }

  {
    MPI_Datatype filetype;
    const int gdims[] = {nx, ny, nz};
    const auto lx = inbox.high[0] - inbox.low[0];
    const auto ly = inbox.high[1] - inbox.low[1];
    const auto lz = inbox.high[2] - inbox.low[2];
    const int ldims[] = {lx, ly, lz};
    const auto ox = inbox.low[0];
    const auto oy = inbox.low[1];
    const auto oz = inbox.low[2];
    const int offset[] = {ox, oy, oz};
    MPI_Type_create_subarray(3, gdims, ldims, offset, MPI_ORDER_C, MPI_DOUBLE,
                             &filetype);
    MPI_Type_commit(&filetype);
    const unsigned int disp = 0;

    MPI_File fh;
    const std::string filename = std::to_string(N) + ".bin";
    MPI_File_open(MPI_COMM_WORLD, filename.c_str(),
                  MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);
    MPI_Offset filesize = 0;
    MPI_File_set_size(fh, filesize); // force overwriting existing data
    MPI_File_set_view(fh, disp, MPI_DOUBLE, filetype, "native", MPI_INFO_NULL);
    MPI_File_write_all(fh, U.data(), U.size(), MPI_DOUBLE, MPI_STATUS_IGNORE);
    MPI_File_close(&fh);
  }
}

int main(int argc, char **argv) {

  argparse::ArgumentParser program("pfc-heffte");

  program.add_argument("settings").help("Simulation settings JSON file");

  try {
    program.parse_args(argc, argv);
  } catch (const std::runtime_error &err) {
    std::cerr << err.what() << std::endl;
    std::cerr << program;
    std::exit(1);
  }

  std::ifstream input_file(program.get<std::string>("settings"));
  json settings;
  input_file >> settings;

  MPI_Init(&argc, &argv);

  compute_dft(MPI_COMM_WORLD, settings);

  MPI_Finalize();

  return 0;
}
