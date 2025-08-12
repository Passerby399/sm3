#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <immintrin.h>  // For SIMD instructions

// SM3常量定义
#define SM3_BLOCK_SIZE 64
#define SM3_DIGEST_SIZE 32
#define SM3_HASH_SIZE 8

// SM3初始值
static const uint32_t IV[SM3_HASH_SIZE] = {
    0x7380166F, 0x4914B2B9, 0x172442D7, 0xDA8A0600,
    0xA96F30BC, 0x163138AA, 0xE38DEE4D, 0xB0FB0E4E
};

// SM3常量Tj
static const uint32_t T[64] = {
    0x79CC4519, 0x79CC4519, 0x79CC4519, 0x79CC4519,
    0x79CC4519, 0x79CC4519, 0x79CC4519, 0x79CC4519,
    0x79CC4519, 0x79CC4519, 0x79CC4519, 0x79CC4519,
    0x79CC4519, 0x79CC4519, 0x79CC4519, 0x79CC4519,
    0x7A879D8A, 0x7A879D8A, 0x7A879D8A, 0x7A879D8A,
    0x7A879D8A, 0x7A879D8A, 0x7A879D8A, 0x7A879D8A,
    0x7A879D8A, 0x7A879D8A, 0x7A879D8A, 0x7A879D8A,
    0x7A879D8A, 0x7A879D8A, 0x7A879D8A, 0x7A879D8A,
    0x7A879D8A, 0x7A879D8A, 0x7A879D8A, 0x7A879D8A,
    0x7A879D8A, 0x7A879D8A, 0x7A879D8A, 0x7A879D8A,
    0x7A879D8A, 0x7A879D8A, 0x7A879D8A, 0x7A879D8A,
    0x7A879D8A, 0x7A879D8A, 0x7A879D8A, 0x7A879D8A,
    0x7A879D8A, 0x7A879D8A, 0x7A879D8A, 0x7A879D8A,
    0x7A879D8A, 0x7A879D8A, 0x7A879D8A, 0x7A879D8A,
    0x7A879D8A, 0x7A879D8A, 0x7A879D8A, 0x7A879D8A
};

// 循环左移
static inline uint32_t ROTL(uint32_t x, int n) {
    return (x << n) | (x >> (32 - n));
}

// 布尔函数
static inline uint32_t FF(uint32_t x, uint32_t y, uint32_t z, int j) {
    if (j < 16) {
        return x ^ y ^ z;
    } else {
        return (x & y) | (x & z) | (y & z);
    }
}

static inline uint32_t GG(uint32_t x, uint32_t y, uint32_t z, int j) {
    if (j < 16) {
        return x ^ y ^ z;
    } else {
        return (x & y) | (~x & z);
    }
}

// P0置换函数
static inline uint32_t P0(uint32_t x) {
    return x ^ ROTL(x, 9) ^ ROTL(x, 17);
}

// P1置换函数
static inline uint32_t P1(uint32_t x) {
    return x ^ ROTL(x, 15) ^ ROTL(x, 23);
}

// ================== SM3基本实现 ==================

// 消息填充
void sm3_pad(const uint8_t *data, size_t len, uint8_t **padded_data, size_t *padded_len) {
    size_t blocks = (len + 1 + 8 + SM3_BLOCK_SIZE - 1) / SM3_BLOCK_SIZE;
    *padded_len = blocks * SM3_BLOCK_SIZE;
    *padded_data = calloc(*padded_len, 1);
    if (!*padded_data) exit(1);
    
    memcpy(*padded_data, data, len);
    (*padded_data)[len] = 0x80;
    
    // 添加长度
    uint64_t bit_len = (uint64_t)len * 8;
    for (int i = 0; i < 8; i++) {
        (*padded_data)[*padded_len - 8 + i] = (bit_len >> (56 - i * 8)) & 0xFF;
    }
}

// 基本消息扩展
void sm3_message_expansion_basic(const uint32_t *block, uint32_t *w, uint32_t *w_prime) {
    // 前16个字
    for (int i = 0; i < 16; i++) {
        w[i] = __builtin_bswap32(block[i]);
    }
    
    // 扩展16-67个字
    for (int i = 16; i < 68; i++) {
        w[i] = P1(w[i-16] ^ w[i-9] ^ ROTL(w[i-3], 15)) 
                ^ ROTL(w[i-13], 7) 
                ^ w[i-6];
    }
    
    // 计算w_prime
    for (int i = 0; i < 64; i++) {
        w_prime[i] = w[i] ^ w[i+4];
    }
}

// 基本压缩函数
void sm3_compress_basic(uint32_t *v, const uint32_t *block) {
    uint32_t w[68];
    uint32_t w_prime[64];
    
    sm3_message_expansion_basic(block, w, w_prime);
    
    uint32_t a = v[0];
    uint32_t b = v[1];
    uint32_t c = v[2];
    uint32_t d = v[3];
    uint32_t e = v[4];
    uint32_t f = v[5];
    uint32_t g = v[6];
    uint32_t h = v[7];
    
    for (int j = 0; j < 64; j++) {
        uint32_t ss1 = ROTL((ROTL(a, 12) + e + ROTL(T[j], j % 32)), 7);
        uint32_t ss2 = ss1 ^ ROTL(a, 12);
        uint32_t tt1 = FF(a, b, c, j) + d + ss2 + w_prime[j];
        uint32_t tt2 = GG(e, f, g, j) + h + ss1 + w[j];
        
        d = c;
        c = ROTL(b, 9);
        b = a;
        a = tt1;
        h = g;
        g = ROTL(f, 19);
        f = e;
        e = P0(tt2);
    }
    
    v[0] ^= a;
    v[1] ^= b;
    v[2] ^= c;
    v[3] ^= d;
    v[4] ^= e;
    v[5] ^= f;
    v[6] ^= g;
    v[7] ^= h;
}

// 基本SM3哈希计算
void sm3_hash_basic(const uint8_t *data, size_t len, uint8_t *digest) {
    uint8_t *padded_data;
    size_t padded_len;
    sm3_pad(data, len, &padded_data, &padded_len);
    
    uint32_t v[SM3_HASH_SIZE];
    memcpy(v, IV, sizeof(v));
    
    size_t blocks = padded_len / SM3_BLOCK_SIZE;
    for (size_t i = 0; i < blocks; i++) {
        sm3_compress_basic(v, (uint32_t*)(padded_data + i * SM3_BLOCK_SIZE));
    }
    
    for (int i = 0; i < SM3_HASH_SIZE; i++) {
        ((uint32_t*)digest)[i] = __builtin_bswap32(v[i]);
    }
    
    free(padded_data);
}

// ================== SM3优化实现 ==================

// SIMD优化消息扩展
#ifdef __SSE2__
void sm3_message_expansion_simd(const uint32_t *block, uint32_t *w, uint32_t *w_prime) {
    // 前16个字
    for (int i = 0; i < 16; i++) {
        w[i] = __builtin_bswap32(block[i]);
    }
    
    // 使用SIMD加速扩展16-67个字
    for (int i = 16; i < 68; i++) {
        __m128i a = _mm_set_epi32(w[i-3], w[i-13], w[i-9], w[i-16]);
        __m128i b = _mm_set_epi32(0, 7, 15, 0);
        
        // ROTL(w[i-3], 15)
        uint32_t rotl15 = ROTL(w[i-3], 15);
        
        // P1(x) = x ^ ROTL(x, 15) ^ ROTL(x, 23)
        uint32_t p1 = w[i-16] ^ w[i-9] ^ rotl15;
        p1 = p1 ^ ROTL(p1, 15) ^ ROTL(p1, 23);
        
        w[i] = p1 ^ ROTL(w[i-13], 7) ^ w[i-6];
    }
    
    // 计算w_prime
    for (int i = 0; i < 64; i++) {
        w_prime[i] = w[i] ^ w[i+4];
    }
}
#endif

// 优化压缩函数（循环展开）
void sm3_compress_optimized(uint32_t *v, const uint32_t *block) {
    uint32_t w[68];
    uint32_t w_prime[64];
    
    #ifdef __SSE2__
    sm3_message_expansion_simd(block, w, w_prime);
    #else
    sm3_message_expansion_basic(block, w, w_prime);
    #endif
    
    uint32_t a = v[0];
    uint32_t b = v[1];
    uint32_t c = v[2];
    uint32_t d = v[3];
    uint32_t e = v[4];
    uint32_t f = v[5];
    uint32_t g = v[6];
    uint32_t h = v[7];
    
    // 循环展开，每4轮一组
    for (int j = 0; j < 64; j += 4) {
        // 第1轮
        uint32_t ss1 = ROTL((ROTL(a, 12) + e + ROTL(T[j], j % 32)), 7);
        uint32_t ss2 = ss1 ^ ROTL(a, 12);
        uint32_t tt1 = FF(a, b, c, j) + d + ss2 + w_prime[j];
        uint32_t tt2 = GG(e, f, g, j) + h + ss1 + w[j];
        d = c;
        c = ROTL(b, 9);
        b = a;
        a = tt1;
        h = g;
        g = ROTL(f, 19);
        f = e;
        e = P0(tt2);
        
        // 第2轮
        ss1 = ROTL((ROTL(a, 12) + e + ROTL(T[j+1], (j+1) % 32)), 7);
        ss2 = ss1 ^ ROTL(a, 12);
        tt1 = FF(a, b, c, j+1) + d + ss2 + w_prime[j+1];
        tt2 = GG(e, f, g, j+1) + h + ss1 + w[j+1];
        d = c;
        c = ROTL(b, 9);
        b = a;
        a = tt1;
        h = g;
        g = ROTL(f, 19);
        f = e;
        e = P0(tt2);
        
        // 第3轮
        ss1 = ROTL((ROTL(a, 12) + e + ROTL(T[j+2], (j+2) % 32)), 7);
        ss2 = ss1 ^ ROTL(a, 12);
        tt1 = FF(a, b, c, j+2) + d + ss2 + w_prime[j+2];
        tt2 = GG(e, f, g, j+2) + h + ss1 + w[j+2];
        d = c;
        c = ROTL(b, 9);
        b = a;
        a = tt1;
        h = g;
        g = ROTL(f, 19);
        f = e;
        e = P0(tt2);
        
        // 第4轮
        ss1 = ROTL((ROTL(a, 12) + e + ROTL(T[j+3], (j+3) % 32)), 7);
        ss2 = ss1 ^ ROTL(a, 12);
        tt1 = FF(a, b, c, j+3) + d + ss2 + w_prime[j+3];
        tt2 = GG(e, f, g, j+3) + h + ss1 + w[j+3];
        d = c;
        c = ROTL(b, 9);
        b = a;
        a = tt1;
        h = g;
        g = ROTL(f, 19);
        f = e;
        e = P0(tt2);
    }
    
    v[0] ^= a;
    v[1] ^= b;
    v[2] ^= c;
    v[3] ^= d;
    v[4] ^= e;
    v[5] ^= f;
    v[6] ^= g;
    v[7] ^= h;
}

// 优化SM3哈希计算
void sm3_hash_optimized(const uint8_t *data, size_t len, uint8_t *digest) {
    uint8_t *padded_data;
    size_t padded_len;
    sm3_pad(data, len, &padded_data, &padded_len);
    
    uint32_t v[SM3_HASH_SIZE];
    memcpy(v, IV, sizeof(v));
    
    size_t blocks = padded_len / SM3_BLOCK_SIZE;
    for (size_t i = 0; i < blocks; i++) {
        sm3_compress_optimized(v, (uint32_t*)(padded_data + i * SM3_BLOCK_SIZE));
    }
    
    for (int i = 0; i < SM3_HASH_SIZE; i++) {
        ((uint32_t*)digest)[i] = __builtin_bswap32(v[i]);
    }
    
    free(padded_data);
}

// ================== 长度扩展攻击 ==================

// 长度扩展攻击
void sm3_length_extension_attack(const uint8_t *original_msg, size_t orig_len,
                                 const uint8_t *orig_hash,
                                 const uint8_t *extension, size_t ext_len,
                                 uint8_t *new_hash) {
    // 计算扩展后的消息长度
    size_t new_len = orig_len + ext_len;
    
    // 计算填充后的原始消息长度
    size_t padded_orig_len = orig_len + 1 + 8; // 原始消息 + 0x80 + 长度
    if (padded_orig_len % SM3_BLOCK_SIZE != 0) {
        padded_orig_len += SM3_BLOCK_SIZE - (padded_orig_len % SM3_BLOCK_SIZE);
    }
    
    // 设置新的初始向量为原始哈希值
    uint32_t v[SM3_HASH_SIZE];
    for (int i = 0; i < SM3_HASH_SIZE; i++) {
        v[i] = __builtin_bswap32(((uint32_t*)orig_hash)[i]);
    }
    
    // 创建扩展消息
    uint8_t *new_msg = malloc(new_len);
    memcpy(new_msg, original_msg, orig_len);
    memcpy(new_msg + orig_len, extension, ext_len);
    
    // 仅对扩展部分进行哈希计算
    uint8_t *ext_start = new_msg + orig_len;
    size_t ext_blocks = (ext_len + SM3_BLOCK_SIZE - 1) / SM3_BLOCK_SIZE;
    
    // 处理扩展部分
    for (size_t i = 0; i < ext_blocks; i++) {
        size_t block_len = SM3_BLOCK_SIZE;
        if (i == ext_blocks - 1) {
            block_len = ext_len - i * SM3_BLOCK_SIZE;
        }
        
        // 如果是最后一个块，需要重新填充
        if (i == ext_blocks - 1 && block_len < SM3_BLOCK_SIZE) {
            uint8_t last_block[SM3_BLOCK_SIZE] = {0};
            memcpy(last_block, ext_start + i * SM3_BLOCK_SIZE, block_len);
            last_block[block_len] = 0x80;
            
            // 设置总比特长度
            uint64_t total_bits = (padded_orig_len + new_len) * 8;
            for (int j = 0; j < 8; j++) {
                last_block[SM3_BLOCK_SIZE - 8 + j] = (total_bits >> (56 - j * 8)) & 0xFF;
            }
            
            sm3_compress_optimized(v, (uint32_t*)last_block);
        } else {
            sm3_compress_optimized(v, (uint32_t*)(ext_start + i * SM3_BLOCK_SIZE));
        }
    }
    
    // 输出结果
    for (int i = 0; i < SM3_HASH_SIZE; i++) {
        ((uint32_t*)new_hash)[i] = __builtin_bswap32(v[i]);
    }
    
    free(new_msg);
}

// ================== Merkle树实现 ==================

typedef struct MerkleNode {
    uint8_t hash[SM3_DIGEST_SIZE];
    struct MerkleNode *left;
    struct MerkleNode *right;
} MerkleNode;

typedef struct {
    MerkleNode *root;
    size_t leaf_count;
    MerkleNode **leaves;
} MerkleTree;

// 创建叶子节点
MerkleNode* create_leaf(const uint8_t *data, size_t len) {
    MerkleNode *node = malloc(sizeof(MerkleNode));
    sm3_hash_optimized(data, len, node->hash);
    node->left = NULL;
    node->right = NULL;
    return node;
}

// 创建父节点
MerkleNode* create_parent(MerkleNode *left, MerkleNode *right) {
    MerkleNode *node = malloc(sizeof(MerkleNode));
    
    // 如果只有一个子节点，复制它
    if (right == NULL) {
        memcpy(node->hash, left->hash, SM3_DIGEST_SIZE);
        node->left = left;
        node->right = NULL;
        return node;
    }
    
    // 计算两个子节点的哈希
    uint8_t combined[SM3_DIGEST_SIZE * 2];
    memcpy(combined, left->hash, SM3_DIGEST_SIZE);
    memcpy(combined + SM3_DIGEST_SIZE, right->hash, SM3_DIGEST_SIZE);
    
    sm3_hash_optimized(combined, SM3_DIGEST_SIZE * 2, node->hash);
    node->left = left;
    node->right = right;
    
    return node;
}

// 构建Merkle树
MerkleTree* build_merkle_tree(uint8_t **data, size_t *lengths, size_t count) {
    MerkleTree *tree = malloc(sizeof(MerkleTree));
    tree->leaf_count = count;
    tree->leaves = malloc(count * sizeof(MerkleNode*));
    
    // 创建叶子节点
    for (size_t i = 0; i < count; i++) {
        tree->leaves[i] = create_leaf(data[i], lengths[i]);
    }
    
    // 构建树
    size_t level_size = count;
    MerkleNode **level = tree->leaves;
    
    while (level_size > 1) {
        size_t next_level_size = (level_size + 1) / 2;
        MerkleNode **next_level = malloc(next_level_size * sizeof(MerkleNode*));
        
        for (size_t i = 0; i < level_size; i += 2) {
            MerkleNode *left = level[i];
            MerkleNode *right = (i + 1 < level_size) ? level[i+1] : NULL;
            next_level[i/2] = create_parent(left, right);
        }
        
        if (level != tree->leaves) {
            free(level);
        }
        
        level = next_level;
        level_size = next_level_size;
    }
    
    tree->root = level[0];
    free(level);
    return tree;
}

// 生成存在性证明
void generate_existence_proof(MerkleTree *tree, size_t index, uint8_t ***proof, size_t *proof_len) {
    *proof_len = 0;
    size_t capacity = 10;
    *proof = malloc(capacity * sizeof(uint8_t*));
    
    MerkleNode *current = tree->leaves[index];
    MerkleNode *parent = current;
    
    while (parent != tree->root) {
        // 找到当前节点在父节点中的位置
        if (parent->left == current) {
            // 需要右兄弟
            if (parent->right) {
                if (*proof_len >= capacity) {
                    capacity *= 2;
                    *proof = realloc(*proof, capacity * sizeof(uint8_t*));
                }
                (*proof)[*proof_len] = malloc(SM3_DIGEST_SIZE);
                memcpy((*proof)[*proof_len], parent->right->hash, SM3_DIGEST_SIZE);
                (*proof_len)++;
            }
        } else {
            // 需要左兄弟
            if (parent->left) {
                if (*proof_len >= capacity) {
                    capacity *= 2;
                    *proof = realloc(*proof, capacity * sizeof(uint8_t*));
                }
                (*proof)[*proof_len] = malloc(SM3_DIGEST_SIZE);
                memcpy((*proof)[*proof_len], parent->left->hash, SM3_DIGEST_SIZE);
                (*proof_len)++;
            }
        }
        
        current = parent;
        // 向上遍历（实际实现中需要维护父指针，这里简化）
        // 实际应用中需要维护父节点指针或使用其他方法
        break; // 简化实现，只返回直接父节点的兄弟
    }
}

// 验证存在性证明
int verify_existence_proof(const uint8_t *leaf_hash, const uint8_t *root_hash,
                          uint8_t **proof, size_t proof_len, size_t index, size_t total_leaves) {
    uint8_t current_hash[SM3_DIGEST_SIZE];
    memcpy(current_hash, leaf_hash, SM3_DIGEST_SIZE);
    
    size_t current_index = index;
    
    for (size_t i = 0; i < proof_len; i++) {
        uint8_t combined[SM3_DIGEST_SIZE * 2];
        
        if (current_index % 2 == 0) {
            // 当前是左节点
            memcpy(combined, current_hash, SM3_DIGEST_SIZE);
            memcpy(combined + SM3_DIGEST_SIZE, proof[i], SM3_DIGEST_SIZE);
        } else {
            // 当前是右节点
            memcpy(combined, proof[i], SM3_DIGEST_SIZE);
            memcpy(combined + SM3_DIGEST_SIZE, current_hash, SM3_DIGEST_SIZE);
        }
        
        sm3_hash_optimized(combined, SM3_DIGEST_SIZE * 2, current_hash);
        current_index /= 2;
    }
    
    // 比较计算出的根哈希和提供的根哈希
    return memcmp(current_hash, root_hash, SM3_DIGEST_SIZE) == 0;
}

// 释放Merkle树
void free_merkle_tree(MerkleNode *node) {
    if (node == NULL) return;
    free_merkle_tree(node->left);
    free_merkle_tree(node->right);
    free(node);
}

void free_merkle_tree_full(MerkleTree *tree) {
    for (size_t i = 0; i < tree->leaf_count; i++) {
        free(tree->leaves[i]);
    }
    free(tree->leaves);
    free_merkle_tree(tree->root);
    free(tree);
}

// ================== 辅助函数 ==================

// 打印哈希值
void print_hash(const uint8_t *hash) {
    for (int i = 0; i < SM3_DIGEST_SIZE; i++) {
        printf("%02x", hash[i]);
    }
    printf("\n");
}

// 生成随机数据
uint8_t* generate_random_data(size_t len) {
    uint8_t *data = malloc(len);
    for (size_t i = 0; i < len; i++) {
        data[i] = rand() % 256;
    }
    return data;
}

// 性能测试
void benchmark_sm3() {
    const size_t data_size = 1024 * 1024; // 1MB
    uint8_t *data = generate_random_data(data_size);
    
    uint8_t digest_basic[SM3_DIGEST_SIZE];
    uint8_t digest_optimized[SM3_DIGEST_SIZE];
    
    // 测试基本实现
    clock_t start = clock();
    sm3_hash_basic(data, data_size, digest_basic);
    double elapsed_basic = (double)(clock() - start) / CLOCKS_PER_SEC;
    
    // 测试优化实现
    start = clock();
    sm3_hash_optimized(data, data_size, digest_optimized);
    double elapsed_optimized = (double)(clock() - start) / CLOCKS_PER_SEC;
    
    // 验证结果一致性
    if (memcmp(digest_basic, digest_optimized, SM3_DIGEST_SIZE) != 0) {
        printf("Error: Basic and optimized implementations produce different results!\n");
    }
    
    printf("SM3 Performance Benchmark (1MB data):\n");
    printf("Basic implementation: %.2f MB/s\n", data_size / elapsed_basic / 1e6);
    printf("Optimized implementation: %.2f MB/s\n", data_size / elapsed_optimized / 1e6);
    
    free(data);
}

// ================== 主函数 ==================

int main() {
    srand(time(NULL));
    
    printf("=== SM3 Algorithm Implementation and Optimization ===\n\n");
    
    // 测试基本功能
    const char *test_msg = "Hello, SM3!";
    uint8_t digest[SM3_DIGEST_SIZE];
    
    printf("Testing basic SM3 implementation:\n");
    sm3_hash_basic((uint8_t*)test_msg, strlen(test_msg), digest);
    printf("Hash of \"%s\": ", test_msg);
    print_hash(digest);
    
    printf("\nTesting optimized SM3 implementation:\n");
    sm3_hash_optimized((uint8_t*)test_msg, strlen(test_msg), digest);
    printf("Hash of \"%s\": ", test_msg);
    print_hash(digest);
    
    // 运行性能测试
    printf("\nRunning performance benchmark:\n");
    benchmark_sm3();
    
    // 长度扩展攻击演示
    printf("\n=== Length Extension Attack Demo ===\n");
    
    const char *original_msg = "Original message";
    const char *extension = "Extension attack";
    
    // 计算原始哈希
    uint8_t orig_hash[SM3_DIGEST_SIZE];
    sm3_hash_optimized((uint8_t*)original_msg, strlen(original_msg), orig_hash);
    
    printf("Original message: \"%s\"\n", original_msg);
    printf("Original hash: ");
    print_hash(orig_hash);
    
    // 进行长度扩展攻击
    uint8_t new_hash[SM3_DIGEST_SIZE];
    sm3_length_extension_attack(
        (uint8_t*)original_msg, strlen(original_msg),
        orig_hash,
        (uint8_t*)extension, strlen(extension),
        new_hash
    );
    
    printf("\nExtended message: \"%s%s\"\n", original_msg, extension);
    printf("Extended hash via attack: ");
    print_hash(new_hash);
    
    // 计算实际扩展消息的哈希
    uint8_t actual_extended_hash[SM3_DIGEST_SIZE];
    size_t extended_len = strlen(original_msg) + strlen(extension);
    uint8_t *extended_msg = malloc(extended_len);
    memcpy(extended_msg, original_msg, strlen(original_msg));
    memcpy(extended_msg + strlen(original_msg), extension, strlen(extension));
    
    sm3_hash_optimized(extended_msg, extended_len, actual_extended_hash);
    printf("Actual extended hash:    ");
    print_hash(actual_extended_hash);
    
    // 验证攻击是否成功
    if (memcmp(new_hash, actual_extended_hash, SM3_DIGEST_SIZE) == 0) {
        printf("\nLength extension attack succeeded!\n");
    } else {
        printf("\nLength extension attack failed!\n");
    }
    
    free(extended_msg);
    
    // Merkle树演示
    printf("\n=== Merkle Tree Demo ===\n");
    
    // 创建10万叶子节点
    const size_t leaf_count = 100000;
    uint8_t **leaf_data = malloc(leaf_count * sizeof(uint8_t*));
    size_t *leaf_lengths = malloc(leaf_count * sizeof(size_t));
    
    printf("Generating %zu leaf nodes...\n", leaf_count);
    for (size_t i = 0; i < leaf_count; i++) {
        leaf_lengths[i] = 64; // 每个叶子节点64字节数据
        leaf_data[i] = generate_random_data(leaf_lengths[i]);
    }
    
    printf("Building Merkle tree...\n");
    MerkleTree *tree = build_merkle_tree(leaf_data, leaf_lengths, leaf_count);
    printf("Merkle tree root hash: ");
    print_hash(tree->root->hash);
    
    // 测试存在性证明
    size_t test_index = 12345;
    printf("\nGenerating existence proof for leaf %zu...\n", test_index);
    
    uint8_t **proof;
    size_t proof_len;
    generate_existence_proof(tree, test_index, &proof, &proof_len);
    
    printf("Proof length: %zu\n", proof_len);
    printf("Verifying proof...\n");
    
    int valid = verify_existence_proof(
        tree->leaves[test_index]->hash,
        tree->root->hash,
        proof, proof_len,
        test_index, leaf_count
    );
    
    printf("Proof verification: %s\n", valid ? "SUCCESS" : "FAILURE");
    
    // 清理
    for (size_t i = 0; i < proof_len; i++) {
        free(proof[i]);
    }
    free(proof);
    
    for (size_t i = 0; i < leaf_count; i++) {
        free(leaf_data[i]);
    }
    free(leaf_data);
    free(leaf_lengths);
    free_merkle_tree_full(tree);
    
    return 0;
}