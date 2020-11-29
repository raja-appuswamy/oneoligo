#include "onejoin.hpp"

namespace logging = boost::log;
namespace keywords = boost::log::keywords;
using namespace std;

void init_logging(bool debug) {
  logging::register_simple_formatter_factory<logging::trivial::severity_level,
                                             char>("Severity");

  logging::add_console_log(
      std::cout, keywords::format = "[%TimeStamp%] [%Severity%] %Message%");

  if (debug) {
    logging::core::get()->set_filter(logging::trivial::severity >=
                                     logging::trivial::debug);
  } else {
    logging::core::get()->set_filter(logging::trivial::severity >=
                                     logging::trivial::info);
  }

  logging::add_common_attributes();
};

void print_configuration(int batch_size, int n_batches, size_t len_output,
                         size_t num_input_strings, int countfilter,
                         int samplingrange) {

  BOOST_LOG_TRIVIAL(info) << std::left << std::setw(50)
                          << "Parameter selected:";
  BOOST_LOG_TRIVIAL(info) << std::left << std::setw(50)
                          << "\tNum of strings:" << num_input_strings;
  BOOST_LOG_TRIVIAL(info) << std::left << std::setw(50)
                          << "\tLen output:" << len_output;
  BOOST_LOG_TRIVIAL(info) << std::left << std::setw(50)
                          << "\tSamplingrange:" << samplingrange;
  BOOST_LOG_TRIVIAL(info) << std::left << std::setw(50)
                          << "\tNumber of Hash Function:" << NUM_HASH;
  BOOST_LOG_TRIVIAL(info) << std::left << std::setw(50)
                          << "\tNumber of Bits per hash function:" << NUM_BITS;
  BOOST_LOG_TRIVIAL(info) << std::left << std::setw(50)
                          << "\tNumber of Random Strings per input string:"
                          << NUM_STR;
  BOOST_LOG_TRIVIAL(info) << std::left << std::setw(50)
                          << "\tNumber of Replication per input string:"
                          << NUM_REP;
  BOOST_LOG_TRIVIAL(info) << std::left << std::setw(50)
                          << "\tK distance:" << K_INPUT;
  BOOST_LOG_TRIVIAL(info) << std::left << std::setw(50)
                          << "\tCount filter:" << countfilter;
  BOOST_LOG_TRIVIAL(info) << std::left << std::setw(50)
                          << "\tBatch size:" << batch_size;
  BOOST_LOG_TRIVIAL(info) << std::left << std::setw(50)
                          << "\tNumber of batches:" << n_batches << std::endl
                          << std::endl;
};

std::string getReportFileName(int device, size_t batch_size) {

  std::string report_name = "";
  if (device == cpu) {
    report_name += "-CPU-";
  } else if (device == gpu) {
    report_name += "-GPU-";
  } else if (device == both) {
    report_name += "-BOTH-";
  } else {
    report_name += "-ERROR";
  }
  report_name += std::to_string(batch_size);
  return report_name;
};

void read_dataset(vector<string> &input_data, string filename) {

  BOOST_LOG_TRIVIAL(info) << "Reading dataset..." << std::endl;
  ifstream data(filename);
  if (!data.is_open()) {
    BOOST_LOG_TRIVIAL(error) << "Error opening input file" << std::endl;
    exit(-1);
  }

  string cell;
  int number_string = 0;
  while (getline(data, cell)) {
    number_string++;
    input_data.push_back(cell);
  }
};

void save_report(int device, size_t batch_size, string dataset_name,
                 OutputValues &output_val, Time &timer) {
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
}