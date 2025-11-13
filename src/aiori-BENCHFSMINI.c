/*
 * Minimal BENCHFS IOR adapter for debugging
 *
 * This is a minimal implementation focusing only on essential IOR operations.
 * This adapter acts as a CLIENT ONLY - servers must be started separately.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <mpi.h>
#include "aiori.h"
#include "iordef.h"

/* FFI declarations */
extern int benchfs_mini_init(const char* registry_dir, int is_server);
extern int benchfs_mini_connect(int server_rank);
extern int benchfs_mini_write(const char* path, const unsigned char* data, size_t data_len, int server_rank);
extern int benchfs_mini_read(const char* path, unsigned char* buffer, size_t buffer_len, int server_rank);
extern int benchfs_mini_finalize(void);
extern void benchfs_mini_progress(void);

#define BENCHFS_MINI_SUCCESS 0
#define BENCHFS_MINI_ERROR -1
#define NUM_SERVERS 2  /* Number of separate server processes */

/* Global state */
static int g_rank = -1;
static int g_size = -1;
static int g_initialized = 0;
static char g_registry_dir[1024] = "/shared/registry_mini";

/* Simple file handle (just stores path and target server) */
typedef struct {
    char path[1024];
    int server_rank;
} benchfs_mini_file_t;

/*
 * Initialize BENCHFS Mini (CLIENT ONLY)
 */
static aiori_fd_t* BENCHFSMINI_Create(char *testFileName, int iorflags, aiori_mod_opt_t *module_options) {
    if (!g_initialized) {
        MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &g_size);

        fprintf(stderr, "[IOR Client Rank %d/%d] Initializing BENCHFS Mini client\n", g_rank, g_size);

        /* Initialize as CLIENT only (is_server = false) */
        int ret = benchfs_mini_init(g_registry_dir, 0);
        if (ret != BENCHFS_MINI_SUCCESS) {
            fprintf(stderr, "[IOR Client Rank %d] Failed to initialize BENCHFS Mini\n", g_rank);
            return NULL;
        }

        /* Wait a bit for servers to register */
        sleep(2);

        /* Connect to all servers */
        fprintf(stderr, "[IOR Client Rank %d] Connecting to %d servers\n", g_rank, NUM_SERVERS);
        for (int i = 0; i < NUM_SERVERS; i++) {
            ret = benchfs_mini_connect(i);
            if (ret != BENCHFS_MINI_SUCCESS) {
                fprintf(stderr, "[IOR Client Rank %d] Failed to connect to server %d\n", g_rank, i);
                return NULL;
            }
            fprintf(stderr, "[IOR Client Rank %d] Connected to server %d\n", g_rank, i);
        }

        /* Sync all client ranks */
        MPI_Barrier(MPI_COMM_WORLD);

        g_initialized = 1;
        fprintf(stderr, "[IOR Client Rank %d] BENCHFS Mini client initialized successfully\n", g_rank);
    }

    /* Create file handle */
    benchfs_mini_file_t* file = malloc(sizeof(benchfs_mini_file_t));
    if (!file) {
        return NULL;
    }

    strncpy(file->path, testFileName, sizeof(file->path) - 1);

    /* Simple hash to determine target server */
    unsigned int hash = 0;
    for (const char* p = testFileName; *p; p++) {
        hash = hash * 31 + *p;
    }
    file->server_rank = hash % NUM_SERVERS;

    fprintf(stderr, "[IOR Client Rank %d] Created file %s mapped to server %d\n",
            g_rank, testFileName, file->server_rank);

    /* Progress UCX */
    for (int i = 0; i < 10; i++) {
        benchfs_mini_progress();
    }

    return (aiori_fd_t*)file;
}

/*
 * Open file
 */
static aiori_fd_t* BENCHFSMINI_Open(char *testFileName, int iorflags, aiori_mod_opt_t *module_options) {
    fprintf(stderr, "[IOR Client Rank %d] Opening file: %s\n", g_rank, testFileName);

    benchfs_mini_file_t* file = malloc(sizeof(benchfs_mini_file_t));
    if (!file) {
        return NULL;
    }

    strncpy(file->path, testFileName, sizeof(file->path) - 1);

    /* Simple hash to determine target server */
    unsigned int hash = 0;
    for (const char* p = testFileName; *p; p++) {
        hash = hash * 31 + *p;
    }
    file->server_rank = hash % NUM_SERVERS;

    fprintf(stderr, "[IOR Client Rank %d] File %s mapped to server %d\n",
            g_rank, testFileName, file->server_rank);

    /* Progress UCX */
    for (int i = 0; i < 10; i++) {
        benchfs_mini_progress();
    }

    return (aiori_fd_t*)file;
}

/*
 * Write/Read data
 */
static IOR_offset_t BENCHFSMINI_Xfer(int access, aiori_fd_t* fd, IOR_size_t* buffer,
                                      IOR_offset_t length, IOR_offset_t offset,
                                      aiori_mod_opt_t* module_options) {
    benchfs_mini_file_t* file = (benchfs_mini_file_t*)fd;

    if (access == WRITE) {
        fprintf(stderr, "[IOR Client Rank %d] Writing %lld bytes to %s (server %d)\n",
                g_rank, (long long)length, file->path, file->server_rank);

        int ret = benchfs_mini_write(file->path, (unsigned char*)buffer,
                                      length, file->server_rank);

        /* Progress UCX */
        for (int i = 0; i < 10; i++) {
            benchfs_mini_progress();
        }

        if (ret != BENCHFS_MINI_SUCCESS) {
            fprintf(stderr, "[IOR Client Rank %d] Write failed\n", g_rank);
            return -1;
        }

        fprintf(stderr, "[IOR Client Rank %d] Write successful\n", g_rank);
        return length;
    } else {
        fprintf(stderr, "[IOR Client Rank %d] Reading %lld bytes from %s (server %d)\n",
                g_rank, (long long)length, file->path, file->server_rank);

        int ret = benchfs_mini_read(file->path, (unsigned char*)buffer,
                                     length, file->server_rank);

        /* Progress UCX */
        for (int i = 0; i < 10; i++) {
            benchfs_mini_progress();
        }

        if (ret < 0) {
            fprintf(stderr, "[IOR Client Rank %d] Read failed\n", g_rank);
            return -1;
        }

        fprintf(stderr, "[IOR Client Rank %d] Read %d bytes successfully\n", g_rank, ret);
        return ret;
    }
}

/*
 * Close file
 */
static void BENCHFSMINI_Close(aiori_fd_t* fd, aiori_mod_opt_t* module_options) {
    benchfs_mini_file_t* file = (benchfs_mini_file_t*)fd;

    fprintf(stderr, "[IOR Client Rank %d] Closing file: %s\n", g_rank, file->path);

    /* Progress UCX */
    for (int i = 0; i < 10; i++) {
        benchfs_mini_progress();
    }

    free(file);
}

/*
 * Delete file (no-op for simplicity)
 */
static void BENCHFSMINI_Delete(char* testFileName, aiori_mod_opt_t* module_options) {
    fprintf(stderr, "[IOR Client Rank %d] Delete file: %s (no-op)\n", g_rank, testFileName);
}

/*
 * Finalize
 */
static void BENCHFSMINI_Finalize(aiori_mod_opt_t* module_options) {
    if (!g_initialized) {
        return;
    }

    fprintf(stderr, "[IOR Client Rank %d] Finalizing BENCHFS Mini\n", g_rank);

    /* Sync all ranks before finalize */
    MPI_Barrier(MPI_COMM_WORLD);

    benchfs_mini_finalize();

    g_initialized = 0;
    fprintf(stderr, "[IOR Client Rank %d] BENCHFS Mini finalized\n", g_rank);
}

/*
 * Get version
 */
static char* BENCHFSMINI_GetVersion(void) {
    return "BenchFS Mini 0.1.0 (Client)";
}

/*
 * Fsync (no-op)
 */
static void BENCHFSMINI_Fsync(aiori_fd_t* fd, aiori_mod_opt_t* module_options) {
    /* No-op for simplicity */
}

/*
 * Get file size (return 0 for simplicity)
 */
static IOR_offset_t BENCHFSMINI_GetFileSize(aiori_mod_opt_t* module_options, char* testFileName) {
    return 0;
}

/* Register backend */
ior_aiori_t benchfsmini_aiori = {
    .name = "BENCHFSMINI",
    .name_legacy = NULL,
    .create = BENCHFSMINI_Create,
    .mknod = NULL,
    .open = BENCHFSMINI_Open,
    .xfer_hints = NULL,
    .xfer = BENCHFSMINI_Xfer,
    .close = BENCHFSMINI_Close,
    .delete = BENCHFSMINI_Delete,
    .get_version = BENCHFSMINI_GetVersion,
    .fsync = BENCHFSMINI_Fsync,
    .get_file_size = BENCHFSMINI_GetFileSize,
    .statfs = NULL,
    .mkdir = NULL,
    .rmdir = NULL,
    .access = NULL,
    .stat = NULL,
    .initialize = NULL,
    .finalize = BENCHFSMINI_Finalize,
    .rename = NULL,
    .get_options = NULL,
    .check_params = NULL,
    .sync = NULL,
    .enable_mdtest = false,
};
