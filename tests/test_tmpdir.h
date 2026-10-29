/* tests/test_tmpdir.h — where the tests put their scratch files.
 *
 * The tests used to spell "/tmp/..." inline, which is a POSIX assumption; a native Windows build has
 * no /tmp. One helper keeps the choice in one place instead of five #ifdefs.
 */
#ifndef E2_TEST_TMPDIR_H
#define E2_TEST_TMPDIR_H

#include <stdio.h>
#include <stdlib.h>

static inline const char *e2_test_tmpdir(void) {
#if defined(_WIN32)
    const char *t = getenv("TEMP");
    if (!t || !t[0]) t = getenv("TMP");
    return (t && t[0]) ? t : ".";
#else
    return "/tmp";
#endif
}

#endif /* E2_TEST_TMPDIR_H */
