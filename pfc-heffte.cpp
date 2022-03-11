// heFFTe implementation of pfc code

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <argparse/argparse.hpp>
#include <heffte.h>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

const double pi = std::atan(1) * 4;

void write(std::string fn, std::vector<double> &data) {
  std::ofstream file(fn, std::ios::out | std::ios::binary);
  auto nbytes = data.size() * sizeof(double);
  file.write(reinterpret_cast<char *>(&data[0]), nbytes);
  file.close();
}

void write_complex(std::string fn, std::vector<std::complex<double>> &data) {
  std::ofstream file(fn, std::ios::out | std::ios::binary);
  auto nbytes = data.size() * sizeof(std::complex<double>);
  file.write(reinterpret_cast<char *>(&data[0]), nbytes);
  file.close();
}

void compute_dft(MPI_Comm comm, json settings) {
  // unpack simulation and model settings
  const int Lx = settings["Lx"];
  const int Ly = settings["Ly"];
  const int Lz = settings["Lz"];
  const double x0 = settings["x0"];
  const double y0 = settings["y0"];
  const double z0 = settings["z0"];
  const double dx = settings["dx"];
  const double dy = settings["dy"];
  const double dz = settings["dz"];
  const double dt = settings["dt"];
  const double tstart = settings["tstart"];
  const double tend = settings["tend"];
  const int maxiters = settings["maxiters"];

  const double Bl = settings["Bl"];
  const double Bx = settings["Bx"];
  const double p3 = settings["p3"];
  const double p4 = settings["p4"];

  const std::string results_dir = settings["results_dir"];

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
  const int Lx_c = floor(Lx / 2) + 1;
  // the dimension where the data will shrink
  const int r2c_direction = 0;
  // define real doman
  heffte::box3d<> real_indexes({0, 0, 0}, {Lx - 1, Ly - 1, Lz - 1});
  // define complex domain
  heffte::box3d<> complex_indexes({0, 0, 0}, {Lx_c - 1, Ly - 1, Lz - 1});

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

  // Define Laplace operator -k^2
  if (me == 0) {
    std::cout << "Generate Laplace operator" << std::endl;
  }
  std::vector<double> k2(fft.size_outbox());
  auto idx = 0;
  auto fx = 2 * pi / (dx * Lx);
  auto fy = 2 * pi / (dy * Ly);
  auto fz = 2 * pi / (dz * Lz);
  for (auto k = outbox.low[2]; k <= outbox.high[2]; k++) {
    for (auto j = outbox.low[1]; j <= outbox.high[1]; j++) {
      for (auto i = outbox.low[0]; i <= outbox.high[0]; i++) {
        auto kx = i < Lx / 2 ? i * fx : (i - Lx) * fx;
        auto ky = j < Ly / 2 ? j * fy : (j - Ly) * fy;
        auto kz = k < Lz / 2 ? k * fz : (k - Lz) * fz;
        k2[idx] = kx * kx + ky * ky + kz * kz;
        idx += 1;
      }
    }
  }

  {
    MPI_Datatype filetype;
    const int gdims[] = {Lx_c, Ly, Lz};
    const auto lx = outbox.high[0] - outbox.low[0] + 1;
    const auto ly = outbox.high[1] - outbox.low[1] + 1;
    const auto lz = outbox.high[2] - outbox.low[2] + 1;
    const int ldims[] = {lx, ly, lz};
    const auto ox = outbox.low[0];
    const auto oy = outbox.low[1];
    const auto oz = outbox.low[2];
    const int offset[] = {ox, oy, oz};
    MPI_Type_create_subarray(3, gdims, ldims, offset, MPI_ORDER_FORTRAN,
                             MPI_DOUBLE, &filetype);
    MPI_Type_commit(&filetype);

    MPI_File fh;
    const std::string filename = results_dir + "k2.bin";
    MPI_File_open(MPI_COMM_WORLD, filename.c_str(),
                  MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);
    MPI_Offset filesize = 0;
    const unsigned int disp = 0;
    MPI_File_set_size(fh, filesize); // force overwriting existing data
    MPI_File_set_view(fh, disp, MPI_DOUBLE, filetype, "native", MPI_INFO_NULL);
    MPI_File_write_all(fh, k2.data(), k2.size(), MPI_DOUBLE, MPI_STATUS_IGNORE);
    MPI_File_close(&fh);
  }

  if (me == 0) {
    std::cout << "Generate initial condition" << std::endl;
  }
  std::vector<double> u(fft.size_inbox());

#ifdef IC_BOX
  idx = 0;
  for (auto k = inbox.low[2]; k <= inbox.high[2]; k++) {
    for (auto j = inbox.low[1]; j <= inbox.high[1]; j++) {
      for (auto i = inbox.low[0]; i <= inbox.high[0]; i++) {
        auto x = x0 + i * dx;
        auto y = y0 + j * dy;
        auto z = z0 + k * dz;
        auto cx = abs(x) <= 10.0 ? 1.0 : 0.0;
        auto cy = abs(y) <= 10.0 ? 1.0 : 0.0;
        auto cz = abs(z) <= 10.0 ? 1.0 : 0.0;
        u[idx] = cx * cy * cz;
        idx += 1;
      }
    }
  }
#endif

  idx = 0;
  for (auto k = inbox.low[2]; k <= inbox.high[2]; k++) {
    for (auto j = inbox.low[1]; j <= inbox.high[1]; j++) {
      for (auto i = inbox.low[0]; i <= inbox.high[0]; i++) {
        auto x = x0 + i * dx;
        auto y = y0 + j * dy;
        auto z = z0 + k * dz;
        auto s1 = sin(1.0 / 16.0 * 2.0 * pi * x);
        auto s2 = sin(1.0 / 16.0 * 2.0 * pi * y);
        auto s3 = sin(1.0 / 16.0 * 2.0 * pi * z);
        auto e1 = exp(-0.01 * x * x);
        auto e2 = exp(-0.01 * y * y);
        auto e3 = exp(-0.01 * z * z);
        u[idx] = pi * s1 * s2 * s3 * e1 * e2 * e3;
        idx += 1;
      }
    }
  }

  // define workspace to improve performance
  std::vector<std::complex<double>> workspace(fft.size_workspace());
  std::vector<std::complex<double>> UL(fft.size_outbox());
  std::vector<std::complex<double>> UN(fft.size_outbox());

  if (me == 0) {
    std::cout << "Starting simulation\n\n";
  }

  {
    MPI_Datatype filetype;
    const int gdims[] = {Lx, Ly, Lz};
    const auto lx = inbox.high[0] - inbox.low[0];
    const auto ly = inbox.high[1] - inbox.low[1];
    const auto lz = inbox.high[2] - inbox.low[2];
    const int ldims[] = {lx + 1, ly + 1, lz + 1};
    const auto ox = inbox.low[0];
    const auto oy = inbox.low[1];
    const auto oz = inbox.low[2];
    const int offset[] = {ox, oy, oz};
    MPI_Type_create_subarray(3, gdims, ldims, offset, MPI_ORDER_FORTRAN,
                             MPI_DOUBLE, &filetype);
    MPI_Type_commit(&filetype);

    const unsigned int disp = 0;
    MPI_File fh;
    const std::string filename = results_dir + "u0.bin";
    MPI_File_open(MPI_COMM_WORLD, filename.c_str(),
                  MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);
    MPI_Offset filesize = 0;
    MPI_File_set_size(fh, filesize); // force overwriting existing data
    MPI_File_set_view(fh, disp, MPI_DOUBLE, filetype, "native", MPI_INFO_NULL);
    MPI_File_write_all(fh, u.data(), u.size(), MPI_DOUBLE, MPI_STATUS_IGNORE);
    MPI_File_close(&fh);
  }

  // start iterations
  auto n = 0;
  auto t = tstart;
  while (t <= tend) {

    n += 1;
    t += dt;

    // Linear part of solution
    MPI_Barrier(MPI_COMM_WORLD);
    fft.forward(u.data(), UL.data(), workspace.data());
    MPI_Barrier(MPI_COMM_WORLD);

    {
      MPI_Datatype filetype;
      const int gdims[] = {Lx_c, Ly, Lz};
      const auto lx = outbox.high[0] - outbox.low[0] + 1;
      const auto ly = outbox.high[1] - outbox.low[1] + 1;
      const auto lz = outbox.high[2] - outbox.low[2] + 1;
      const int ldims[] = {lx, ly, lz};
      const auto ox = outbox.low[0];
      const auto oy = outbox.low[1];
      const auto oz = outbox.low[2];
      const int offset[] = {ox, oy, oz};
      MPI_Type_create_subarray(3, gdims, ldims, offset, MPI_ORDER_FORTRAN,
                               MPI_DOUBLE_COMPLEX, &filetype);
      MPI_Type_commit(&filetype);

      MPI_File fh;
      const std::string filename =
          results_dir + "UL_" + std::to_string(n) + ".bin";
      MPI_File_open(MPI_COMM_WORLD, filename.c_str(),
                    MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);
      MPI_Offset filesize = 0;
      const unsigned int disp = 0;
      MPI_File_set_size(fh, filesize); // force overwriting existing data
      MPI_File_set_view(fh, disp, MPI_DOUBLE_COMPLEX, filetype, "native",
                        MPI_INFO_NULL);
      MPI_File_write_all(fh, UL.data(), UL.size(), MPI_DOUBLE_COMPLEX,
                         MPI_STATUS_IGNORE);
      MPI_File_close(&fh);
    }

    // Nonlinear part of solution
    for (auto i = 0; i < u.size(); i++) {
      auto ui = u[i];
      u[i] = p3 * ui * ui + p4 * ui * ui * ui;
    }
    fft.forward(u.data(), UN.data(), workspace.data());

    // Semi-implicit time integration, store results to UL
    for (auto i = 0; i < UL.size(); i++) {
      auto k2i = k2[i];
      auto Ci = -Bx * (-2.0 * k2i + k2i * k2i);
      auto Li = -k2i * (Bl - Ci);
      UL[i] = 1.0 / (1.0 - dt * Li) * (UL[i] - k2i * dt * UN[i]);
    };

    // Back to real space
    fft.backward(UL.data(), u.data(), workspace.data(), heffte::scale::full);

    if (me == 0) {
      std::cout << "Iteration " << n << ", t = " << t << std::endl;
    }

    {
      MPI_Datatype filetype;
      const int gdims[] = {Lx, Ly, Lz};
      const auto lx = inbox.high[0] - inbox.low[0];
      const auto ly = inbox.high[1] - inbox.low[1];
      const auto lz = inbox.high[2] - inbox.low[2];
      const int ldims[] = {lx + 1, ly + 1, lz + 1};
      const auto ox = inbox.low[0];
      const auto oy = inbox.low[1];
      const auto oz = inbox.low[2];
      const int offset[] = {ox, oy, oz};
      MPI_Type_create_subarray(3, gdims, ldims, offset, MPI_ORDER_FORTRAN,
                               MPI_DOUBLE, &filetype);
      MPI_Type_commit(&filetype);
      const unsigned int disp = 0;

      MPI_File fh;
      const std::string filename =
          results_dir + "u" + std::to_string(n) + ".bin";
      MPI_File_open(MPI_COMM_WORLD, filename.c_str(),
                    MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);
      MPI_Offset filesize = 0;
      MPI_File_set_size(fh, filesize); // force overwriting existing data
      MPI_File_set_view(fh, disp, MPI_DOUBLE, filetype, "native",
                        MPI_INFO_NULL);
      MPI_File_write_all(fh, u.data(), u.size(), MPI_DOUBLE, MPI_STATUS_IGNORE);
      MPI_File_close(&fh);
    }

    if (n == maxiters) {
      if (me == 0) {
        std::cout << "Maximum number of iterations (" << maxiters
                  << ") reached, exiting" << std::endl;
      }
      break;
    }
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
