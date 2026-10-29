/* <unistd.h> for a native Windows build — see e2win.h.
 *
 * UCRT already has open/close/read/write/unlink/fdopen under their POSIX names; this only adds the
 * two that are genuinely missing (sched_yield, mkstemp) and the fsync spelling.
 */
#ifndef E2W_UNISTD_H
#define E2W_UNISTD_H

#include <fcntl.h>
#include <io.h>
#include <process.h>
#include <sys/stat.h>
#include <sys/types.h>

/* sched_yield lives in the sched.h shim, which is where POSIX puts it. */

/* q8cache.c stages its temporary next to the destination and renames it into place. */
int mkstemp(char *tmpl);

/* UCRT spells it _commit. */
#define fsync _commit

#endif /* E2W_UNISTD_H */
