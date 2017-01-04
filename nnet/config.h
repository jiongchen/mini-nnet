#ifndef MININET_CONFIG_H
#define MININET_CONFIG_H

#define CHECK(x) \
    do { \
      if (!(x)) { \
        std::cerr << "# error: assertion failed at" << std::endl; \
        std::cerr << __FILE__ << " " << __LINE__ << std::endl; \
        exit(0); \
      } \
    } while(0);

#endif
