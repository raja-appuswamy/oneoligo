#include "onejoin.hpp"

int main(int argc, char **argv) {

  namespace po = boost::program_options;
  bool help{};
  po::options_description description("onejoin [options]");
  description.add_options()("help,h", po::bool_switch(&help), "Display help")(
      "alg,a", po::value<int>(), "Algorithm to use: 1-join [default], 2-cluster")(
      "read,r", po::value<string>(), "File containing input strings")(
      "device,d", po::value<int>(), "Device: 0-CPU; 1-GPU; 2-both devices")(
      "samplingrange,s", po::value<uint32_t>(), "Max char to embed")(
      "countfilter,c", po::value<uint32_t>(),
      "Min number of occurrencies for a pair to be considered a candidate")(
      "batch_size,b", po::value<size_t>(),
      "Size of input strings batches")("verbose,v", "[optional] Print debug information")(
      "dataset_name,n", po::value<string>(), "[optional] Name of dataset to use in the report name")(
      "num_thread_ed_dist,t", po::value<int>(),"[optional] Number of thread to use for edit distance. Default 0 (hardware cuncurrency)")(
      "min_pts,p", po::value<int>(), "[optional] Min number of neighbours a point has to have. Default 10"
      );

  po::command_line_parser parser{argc, argv};
  parser.options(description);
  auto parsed_result = parser.run();
  po::variables_map vm;
  po::store(parsed_result, vm);
  po::notify(vm);

  
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
    dataset_name=vm["dataset_name"].as<string>();
  }

  bool debug = false;
  if (vm.count("verbose")) {
    debug = true;
  }

  int num_thread=0;
  if( vm.count("num_thread_ed_dist") ){
    num_thread=vm["num_thread_ed_dist"].as<int>();
  }

  int alg=alg::join;
  if( vm.count("alg") && ( vm["alg"].as<int>()==alg::join || vm["alg"].as<int>()==alg::cluster ) ){
    alg=vm["alg"].as<int>();
  }

  int min_pts=10;
  if( vm.count("min_pts") ){
    min_pts=vm["min_pts"].as<int>();
  }

  init_logging(debug);

  vector<string> input_data;
  read_dataset(input_data, filename);

  OutputValues output_val;

  Time timer((alg==alg::join?false:true));

  if(alg==alg::join){
    onejoin(input_data, batch_size, device, samplingrange, countfilter, timer,
           output_val, alg::join, num_thread);
  }
  else{
   oneCluster(input_data, batch_size, device, samplingrange, countfilter, timer, min_pts, "GEN320");
  }

  save_report( device, batch_size, dataset_name, output_val, timer );
  return 0;
}
