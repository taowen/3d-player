#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>
#include <crtdbg.h>
#include <windows.h>

// Custom report hook to handle assertions and continue execution
int CustomAssertHook(int reportType, char* message, int* returnValue) {
    if (reportType == _CRT_ASSERT) {
        // Print the assertion message to stderr
        if (message) {
            fprintf(stderr, "Debug Assertion Failed: %s\n", message);
        }
        // Continue execution instead of aborting
        *returnValue = 1;
        return TRUE; // We handled it
    }
    return FALSE; // Let default handler process other report types
}

// Redirect CRT assertions to stderr without dialog boxes
static struct CrtAssertionRedirector {
    CrtAssertionRedirector() {
        // Set custom report hook to handle assertions
        _CrtSetReportHook(CustomAssertHook);
        
        // Only redirect to console if no debugger is attached
        if (!IsDebuggerPresent()) {
            _CrtSetReportMode(_CRT_ASSERT, _CRTDBG_MODE_FILE | _CRTDBG_MODE_DEBUG);
            _CrtSetReportFile(_CRT_ASSERT, _CRTDBG_FILE_STDERR);
            _CrtSetReportMode(_CRT_WARN, _CRTDBG_MODE_FILE | _CRTDBG_MODE_DEBUG);
            _CrtSetReportFile(_CRT_WARN, _CRTDBG_FILE_STDERR);
            _CrtSetReportMode(_CRT_ERROR, _CRTDBG_MODE_FILE | _CRTDBG_MODE_DEBUG);
            _CrtSetReportFile(_CRT_ERROR, _CRTDBG_FILE_STDERR);
        }
    }
} crt_assertion_redirector;