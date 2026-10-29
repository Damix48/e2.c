/* <sched.h> for a native Windows build — see e2win.h. threads.c includes it for sched_yield. */
#ifndef E2W_SCHED_H
#define E2W_SCHED_H
int e2w_sched_yield(void);
#define sched_yield() e2w_sched_yield()
#endif
