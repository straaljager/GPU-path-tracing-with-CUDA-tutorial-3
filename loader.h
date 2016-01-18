#ifndef __LOADER_H_
#define __LOADER_H_

void panic(const char *fmt, ...);
float load_object(const char *filename);

#endif
