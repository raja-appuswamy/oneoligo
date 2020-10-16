#include "embedjoin.hpp"
namespace logging = boost::log;
namespace keywords = boost::log::keywords;
using namespace std;

void init_logging()
{
    logging::register_simple_formatter_factory<logging::trivial::severity_level, char>("Severity");

    logging::add_console_log( std::cout, keywords::format = "[%TimeStamp%] [%Severity%] %Message%" );

    logging::core::get()->set_filter
    (
        logging::trivial::severity >= logging::trivial::trace
    );

    logging::add_common_attributes();
};

void print_configuration(int batch_size, int n_batches, size_t len_output,
                         size_t num_input_strings, int countfilter,
                         int samplingrange) {
  std::cout << "\nParameter selected:" << std::endl;
  std::cout << "\tNum of strings:\t\t\t\t\t" << num_input_strings << std::endl;
  std::cout << "\tLen output:\t\t\t\t\t" << len_output << std::endl;
  std::cout << "\tSamplingrange:\t\t\t\t\t" << samplingrange << std::endl;
  std::cout << "\tNumber of Hash Function:\t\t\t" << NUM_HASH << std::endl;
  std::cout << "\tNumber of Bits per hash function:\t\t" << NUM_BITS
            << std::endl;
  std::cout << "\tNumber of Random Strings per input string:\t" << NUM_STR
            << std::endl;
  std::cout << "\tNumber of Replication per input string:\t\t" << NUM_REP
            << std::endl;
  std::cout << "\tK distance:\t\t\t\t\t" << K_INPUT << std::endl;
  std::cout << "\tCount filter:\t\t\t\t\t" << countfilter << std::endl;
  std::cout << "\tBatch size:\t\t\t\t\t" << batch_size << std::endl;
  std::cout << "\tNumber of batches:\t\t\t\t" << n_batches << std::endl;
  std::cout << std::endl;
  std::cout << std::endl;
};

std::string getReportFileName(int device, size_t batch_size) {

  std::string report_name = "";
  if (device == 0) {
    report_name += "-CPU-";
  } else if (device == 1) {
    report_name += "-GPU-";
  } else if (device == 2) {
    report_name += "-BOTH-";
  } else {
    report_name += "-ERROR";
  }
  report_name += std::to_string(batch_size);
  return report_name;
}
