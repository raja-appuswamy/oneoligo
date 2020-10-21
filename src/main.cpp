#include "embedjoin.hpp"

int main(int argc, char **argv) {

  namespace po = boost::program_options;
  bool help{};
  po::options_description description("onejoin [options]");
  description.add_options()("help,h", po::bool_switch(&help), "Display help")(
      "read,r", po::value<string>(), "File containing input strings")(
      "device,d", po::value<int>(), "Device: 0-CPU; 1-GPU; 2-both devices")(
      "samplingrange,s", po::value<uint32_t>(), "Max char to embed")(
      "countfilter,c", po::value<uint32_t>(),
      "Min number of occurrencies for a pair to be considered a candidate")(
      "batch_size,b", po::value<size_t>(),
      "Size of input strings batches")("verbose,v", "[optional] Print debug information")(
      "dataset_name,n", po::value<string>(), "[optional] Name of dataset to use in the report name");

  po::command_line_parser parser{argc, argv};
  parser.options(description);
  auto parsed_result = parser.run();
  po::variables_map vm;
  po::store(parsed_result, vm);
  po::notify(vm);

  Time timer;
  int device = 0;
  size_t batch_size = 0;
  string filename = "";
  uint32_t samplingrange = 0; // The maximum digit to embed, the range to sample
  uint32_t countfilter = 0;   // Number of required matches (>T) for a pair of
                              // substrings to be considered as candidate

  if (help) {
    std::cerr << description << std::endl;
    return 0;
  }

  if (vm.count("read") && vm.count("device") && vm.count("samplingrange") &&
      vm.count("countfilter") && vm.count("batch_size")) {

    filename = vm["read"].as<string>();
    device = vm["device"].as<int>();
    samplingrange = vm["samplingrange"].as<uint32_t>();
    countfilter = vm["countfilter"].as<uint32_t>();
    batch_size = vm["batch_size"].as<size_t>();
  } else {
    std::cerr << description << std::endl;
    return 1;
  }

  string dataset_name="";
  if (vm.count("dataset_name")) {
    dataset_name=vm["batch_size"].as<string>();
  }

  bool debug = false;
  if (vm.count("verbose")) {
    debug = true;
  }

  init_logging(debug);

  vector<string> input_data;
  read_dataset(input_data, filename);
  OutputValues output_val;

  onejoin(input_data, batch_size, device, samplingrange, countfilter, timer,
          output_val);

  string report_name = getReportFileName(device, batch_size);

  {
    ofstream out_file;
    out_file.open("report-" + dataset_name + report_name + ".csv",
                  ios::out | ios::trunc);

    if (out_file.is_open()) {
      timer.print_report(output_val.dev, output_val.num_candidates,
                         output_val.num_outputs, out_file);
    }
  }

  return 0;
}
