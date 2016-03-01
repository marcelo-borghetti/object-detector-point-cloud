#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <iomanip>
#include <ctime>
#include <time.h>
#include <stdio.h>
#include <boost/filesystem.hpp>
#include "../include/mlp_pkg/logsFile.h"
using namespace std; 

void LogsFile::createStructure(string numDir)
{
	//creating diretory for the results
	//this->baseDir << "Results" << "/" << numDir;
	this->baseDir << numDir;
	/*boost::filesystem::path dir(baseDir.str());
	if (!boost::filesystem::create_directory(dir)) {
		std::cout << "Base directory not created" << "\n";
	}*/
	
	//Create the file that will store the log
	this->filename << this->baseDir.str() << "/log.txt";
}

void LogsFile::createDir(string nameDir)
{
	boost::filesystem::path dir(nameDir);
	if (!boost::filesystem::create_directory(dir)) {
		//std::cout << "Base directory not created" << "\n";
	}
}

void LogsFile::add(string str)
{
    time_t     now = time(0);
    struct tm  tstruct;
    char       buf[80];
    tstruct = *localtime(&now);
    // Visit http://en.cppreference.com/w/cpp/chrono/c/strftime
    // for more information about date/time format
    strftime(buf, sizeof(buf), "%Y-%m-%d.%X", &tstruct);
        
    this->myfile.open(this->filename.str().c_str(), ios::out | ios::app | ios::binary);
    myfile << buf << " " << str;
	this->myfile.close(); 
}

