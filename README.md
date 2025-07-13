# Data Structures Reference Guide 📚

*A comprehensive guide to data structures with Kotlin implementations*

## Table of Contents
- [Introduction](#introduction)
- [Linear Data Structures](#linear-data-structures)
  - [Arrays](#arrays)
  - [Linked Lists](#linked-lists)
  - [Stacks](#stacks)
  - [Queues](#queues)
- [Non-Linear Data Structures](#non-linear-data-structures)
  - [Trees](#trees)
  - [Graphs](#graphs)
  - [Hash Tables](#hash-tables)
- [Advanced Data Structures](#advanced-data-structures)
  - [Heaps](#heaps)
  - [Tries](#tries)
- [Time & Space Complexity](#time--space-complexity)
- [Practice Problems](#practice-problems)

---

## Introduction

Data structures are fundamental building blocks in computer science that allow us to organize, store, and manipulate data efficiently. This guide provides practical Kotlin implementations with visual representations and real-world examples.

### Why Learn Data Structures?
- **Efficiency**: Choose the right tool for the job
- **Problem Solving**: Essential for coding interviews and real-world applications
- **Foundation**: Building blocks for algorithms and system design

---

## Linear Data Structures

### Arrays

Arrays store elements in contiguous memory locations with constant-time access by index.

```
Visual: [10] [20] [30] [40] [50]
Index:   0    1    2    3    4
```

#### Implementation
```kotlin
class DynamicArray<T> {
    private var array: Array<Any?> = arrayOfNulls(10)
    private var size = 0
    
    fun add(element: T) {
        if (size >= array.size) {
            resize()
        }
        array[size++] = element
    }
    
    fun get(index: Int): T {
        if (index >= size) throw IndexOutOfBoundsException()
        return array[index] as T
    }
    
    fun remove(index: Int): T {
        if (index >= size) throw IndexOutOfBoundsException()
        val element = array[index] as T
        
        // Shift elements left
        for (i in index until size - 1) {
            array[i] = array[i + 1]
        }
        size--
        array[size] = null
        return element
    }
    
    private fun resize() {
        val newArray = arrayOfNulls<Any?>(array.size * 2)
        for (i in 0 until size) {
            newArray[i] = array[i]
        }
        array = newArray
    }
    
    fun size() = size
    fun isEmpty() = size == 0
}
```

#### Time Complexity
- **Access**: O(1)
- **Search**: O(n)
- **Insertion**: O(n) average, O(1) at end
- **Deletion**: O(n)

#### Use Cases
- When you need fast random access to elements
- Fixed-size collections
- Mathematical operations on sequences

---

### Linked Lists

A linked list consists of nodes where each node contains data and a reference to the next node.

```
Visual: [Data|Next] -> [Data|Next] -> [Data|Next] -> null
```

#### Singly Linked List
```kotlin
class Node<T>(var data: T, var next: Node<T>? = null)

class LinkedList<T> {
    private var head: Node<T>? = null
    private var size = 0
    
    fun addFirst(data: T) {
        val newNode = Node(data, head)
        head = newNode
        size++
    }
    
    fun addLast(data: T) {
        val newNode = Node(data)
        if (head == null) {
            head = newNode
        } else {
            var current = head
            while (current?.next != null) {
                current = current.next
            }
            current?.next = newNode
        }
        size++
    }
    
    fun removeFirst(): T? {
        if (head == null) return null
        val data = head?.data
        head = head?.next
        size--
        return data
    }
    
    fun find(data: T): Boolean {
        var current = head
        while (current != null) {
            if (current.data == data) return true
            current = current.next
        }
        return false
    }
    
    fun display() {
        var current = head
        while (current != null) {
            print("${current.data} -> ")
            current = current.next
        }
        println("null")
    }
    
    fun size() = size
}
```

#### Doubly Linked List
```kotlin
class DoublyNode<T>(
    var data: T,
    var next: DoublyNode<T>? = null,
    var prev: DoublyNode<T>? = null
)

class DoublyLinkedList<T> {
    private var head: DoublyNode<T>? = null
    private var tail: DoublyNode<T>? = null
    private var size = 0
    
    fun addFirst(data: T) {
        val newNode = DoublyNode(data)
        if (head == null) {
            head = newNode
            tail = newNode
        } else {
            newNode.next = head
            head?.prev = newNode
            head = newNode
        }
        size++
    }
    
    fun addLast(data: T) {
        val newNode = DoublyNode(data)
        if (tail == null) {
            head = newNode
            tail = newNode
        } else {
            tail?.next = newNode
            newNode.prev = tail
            tail = newNode
        }
        size++
    }
    
    fun removeFirst(): T? {
        if (head == null) return null
        val data = head?.data
        head = head?.next
        head?.prev = null
        if (head == null) tail = null
        size--
        return data
    }
    
    fun removeLast(): T? {
        if (tail == null) return null
        val data = tail?.data
        tail = tail?.prev
        tail?.next = null
        if (tail == null) head = null
        size--
        return data
    }
}
```

#### Time Complexity
- **Access**: O(n)
- **Search**: O(n)
- **Insertion**: O(1) at head/tail, O(n) at arbitrary position
- **Deletion**: O(1) at head/tail, O(n) at arbitrary position

#### Use Cases
- Dynamic size requirements
- Frequent insertions/deletions
- Implementing other data structures (stacks, queues)

---

### Stacks

A stack follows the Last In, First Out (LIFO) principle.

```
Visual:
    |   30  |  <- Top (last in, first out)
    |   20  |
    |   10  |
    +-------+
```

#### Implementation
```kotlin
class Stack<T> {
    private val items = mutableListOf<T>()
    
    fun push(item: T) {
        items.add(item)
    }
    
    fun pop(): T? {
        return if (items.isEmpty()) null else items.removeAt(items.size - 1)
    }
    
    fun peek(): T? {
        return if (items.isEmpty()) null else items[items.size - 1]
    }
    
    fun isEmpty() = items.isEmpty()
    fun size() = items.size
    
    fun display() {
        println("Stack (top to bottom): ${items.reversed()}")
    }
}

// Usage Example
fun balancedParentheses(expression: String): Boolean {
    val stack = Stack<Char>()
    val pairs = mapOf(')' to '(', '}' to '{', ']' to '[')
    
    for (char in expression) {
        when (char) {
            '(', '{', '[' -> stack.push(char)
            ')', '}', ']' -> {
                if (stack.isEmpty() || stack.pop() != pairs[char]) {
                    return false
                }
            }
        }
    }
    return stack.isEmpty()
}
```

#### Time Complexity
- **Push**: O(1)
- **Pop**: O(1)
- **Peek**: O(1)

#### Use Cases
- Function call management
- Expression evaluation
- Undo operations
- Browser history

---

### Queues

A queue follows the First In, First Out (FIFO) principle.

```
Visual:
Front -> [10] [20] [30] [40] <- Rear
         (dequeue)    (enqueue)
```

#### Implementation
```kotlin
class Queue<T> {
    private val items = mutableListOf<T>()
    
    fun enqueue(item: T) {
        items.add(item)
    }
    
    fun dequeue(): T? {
        return if (items.isEmpty()) null else items.removeAt(0)
    }
    
    fun front(): T? {
        return if (items.isEmpty()) null else items[0]
    }
    
    fun isEmpty() = items.isEmpty()
    fun size() = items.size
    
    fun display() {
        println("Queue (front to rear): $items")
    }
}

// Circular Queue Implementation
class CircularQueue<T>(private val capacity: Int) {
    private val array = arrayOfNulls<Any?>(capacity)
    private var front = 0
    private var rear = -1
    private var size = 0
    
    fun enqueue(item: T): Boolean {
        if (size >= capacity) return false
        rear = (rear + 1) % capacity
        array[rear] = item
        size++
        return true
    }
    
    fun dequeue(): T? {
        if (size == 0) return null
        val item = array[front] as T
        array[front] = null
        front = (front + 1) % capacity
        size--
        return item
    }
    
    fun isFull() = size == capacity
    fun isEmpty() = size == 0
}
```

#### Time Complexity
- **Enqueue**: O(1)
- **Dequeue**: O(1)
- **Front**: O(1)

#### Use Cases
- Process scheduling
- Breadth-first search
- Handling requests in web servers
- Print queue management

---

## Non-Linear Data Structures

### Trees

A tree is a hierarchical data structure with nodes connected by edges.

```
Visual:
        1
       / \
      2   3
     / \   \
    4   5   6
```

#### Binary Tree
```kotlin
class TreeNode<T>(
    var data: T,
    var left: TreeNode<T>? = null,
    var right: TreeNode<T>? = null
)

class BinaryTree<T> {
    var root: TreeNode<T>? = null
    
    // Tree Traversals
    fun inorderTraversal(node: TreeNode<T>? = root, result: MutableList<T> = mutableListOf()): List<T> {
        node?.let {
            inorderTraversal(it.left, result)
            result.add(it.data)
            inorderTraversal(it.right, result)
        }
        return result
    }
    
    fun preorderTraversal(node: TreeNode<T>? = root, result: MutableList<T> = mutableListOf()): List<T> {
        node?.let {
            result.add(it.data)
            preorderTraversal(it.left, result)
            preorderTraversal(it.right, result)
        }
        return result
    }
    
    fun postorderTraversal(node: TreeNode<T>? = root, result: MutableList<T> = mutableListOf()): List<T> {
        node?.let {
            postorderTraversal(it.left, result)
            postorderTraversal(it.right, result)
            result.add(it.data)
        }
        return result
    }
    
    fun levelOrderTraversal(): List<T> {
        val result = mutableListOf<T>()
        val queue = Queue<TreeNode<T>>()
        
        root?.let { queue.enqueue(it) }
        
        while (!queue.isEmpty()) {
            val node = queue.dequeue()!!
            result.add(node.data)
            
            node.left?.let { queue.enqueue(it) }
            node.right?.let { queue.enqueue(it) }
        }
        
        return result
    }
    
    fun height(node: TreeNode<T>? = root): Int {
        return if (node == null) 0
        else 1 + maxOf(height(node.left), height(node.right))
    }
}
```

#### Binary Search Tree
```kotlin
class BST<T : Comparable<T>> {
    var root: TreeNode<T>? = null
    
    fun insert(data: T) {
        root = insertRec(root, data)
    }
    
    private fun insertRec(node: TreeNode<T>?, data: T): TreeNode<T> {
        if (node == null) return TreeNode(data)
        
        when {
            data < node.data -> node.left = insertRec(node.left, data)
            data > node.data -> node.right = insertRec(node.right, data)
        }
        return node
    }
    
    fun search(data: T): Boolean {
        return searchRec(root, data)
    }
    
    private fun searchRec(node: TreeNode<T>?, data: T): Boolean {
        if (node == null) return false
        
        return when {
            data == node.data -> true
            data < node.data -> searchRec(node.left, data)
            else -> searchRec(node.right, data)
        }
    }
    
    fun delete(data: T) {
        root = deleteRec(root, data)
    }
    
    private fun deleteRec(node: TreeNode<T>?, data: T): TreeNode<T>? {
        if (node == null) return null
        
        when {
            data < node.data -> node.left = deleteRec(node.left, data)
            data > node.data -> node.right = deleteRec(node.right, data)
            else -> {
                // Node to be deleted found
                if (node.left == null) return node.right
                if (node.right == null) return node.left
                
                // Node has two children
                val minValue = findMin(node.right!!)
                node.data = minValue
                node.right = deleteRec(node.right, minValue)
            }
        }
        return node
    }
    
    private fun findMin(node: TreeNode<T>): T {
        var current = node
        while (current.left != null) {
            current = current.left!!
        }
        return current.data
    }
}
```

#### Time Complexity (BST)
- **Search**: O(log n) average, O(n) worst case
- **Insertion**: O(log n) average, O(n) worst case
- **Deletion**: O(log n) average, O(n) worst case

#### Use Cases
- Database indexing
- Expression parsing
- File system organization
- Decision trees

---

### Graphs

A graph consists of vertices (nodes) connected by edges.

```
Visual:
    A --- B
    |     |
    |     |
    D --- C
```

#### Graph Representations
```kotlin
// Adjacency List Representation
class Graph<T> {
    private val adjacencyList = mutableMapOf<T, MutableList<T>>()
    
    fun addVertex(vertex: T) {
        adjacencyList.putIfAbsent(vertex, mutableListOf())
    }
    
    fun addEdge(source: T, destination: T) {
        adjacencyList[source]?.add(destination)
        adjacencyList[destination]?.add(source) // For undirected graph
    }
    
    fun removeEdge(source: T, destination: T) {
        adjacencyList[source]?.remove(destination)
        adjacencyList[destination]?.remove(source)
    }
    
    fun getNeighbors(vertex: T): List<T> {
        return adjacencyList[vertex] ?: emptyList()
    }
    
    // Depth-First Search
    fun dfs(start: T, visited: MutableSet<T> = mutableSetOf()): List<T> {
        val result = mutableListOf<T>()
        
        fun dfsHelper(vertex: T) {
            visited.add(vertex)
            result.add(vertex)
            
            adjacencyList[vertex]?.forEach { neighbor ->
                if (neighbor !in visited) {
                    dfsHelper(neighbor)
                }
            }
        }
        
        dfsHelper(start)
        return result
    }
    
    // Breadth-First Search
    fun bfs(start: T): List<T> {
        val result = mutableListOf<T>()
        val visited = mutableSetOf<T>()
        val queue = Queue<T>()
        
        queue.enqueue(start)
        visited.add(start)
        
        while (!queue.isEmpty()) {
            val vertex = queue.dequeue()!!
            result.add(vertex)
            
            adjacencyList[vertex]?.forEach { neighbor ->
                if (neighbor !in visited) {
                    visited.add(neighbor)
                    queue.enqueue(neighbor)
                }
            }
        }
        
        return result
    }
    
    fun display() {
        adjacencyList.forEach { (vertex, neighbors) ->
            println("$vertex -> ${neighbors.joinToString(", ")}")
        }
    }
}
```

#### Weighted Graph
```kotlin
data class Edge<T>(val destination: T, val weight: Int)

class WeightedGraph<T> {
    private val adjacencyList = mutableMapOf<T, MutableList<Edge<T>>>()
    
    fun addVertex(vertex: T) {
        adjacencyList.putIfAbsent(vertex, mutableListOf())
    }
    
    fun addEdge(source: T, destination: T, weight: Int) {
        adjacencyList[source]?.add(Edge(destination, weight))
        adjacencyList[destination]?.add(Edge(source, weight)) // For undirected graph
    }
    
    // Dijkstra's Algorithm for shortest path
    fun dijkstra(start: T): Map<T, Int> {
        val distances = mutableMapOf<T, Int>()
        val visited = mutableSetOf<T>()
        val priorityQueue = java.util.PriorityQueue<Pair<T, Int>> { a, b -> a.second - b.second }
        
        // Initialize distances
        adjacencyList.keys.forEach { vertex ->
            distances[vertex] = Int.MAX_VALUE
        }
        distances[start] = 0
        priorityQueue.offer(Pair(start, 0))
        
        while (priorityQueue.isNotEmpty()) {
            val (current, currentDist) = priorityQueue.poll()
            
            if (current in visited) continue
            visited.add(current)
            
            adjacencyList[current]?.forEach { edge ->
                val newDist = currentDist + edge.weight
                if (newDist < distances[edge.destination]!!) {
                    distances[edge.destination] = newDist
                    priorityQueue.offer(Pair(edge.destination, newDist))
                }
            }
        }
        
        return distances
    }
}
```

#### Time Complexity
- **DFS/BFS**: O(V + E) where V = vertices, E = edges
- **Dijkstra**: O((V + E) log V)

#### Use Cases
- Social networks
- GPS navigation
- Web crawling
- Network routing

---

### Hash Tables

Hash tables provide fast access to data using key-value pairs.

```
Visual:
Index:  0    1    2    3    4
      [  ] [A] [  ] [B] [C]
             |         |    |
           Value1   Value2 Value3
```

#### Implementation
```kotlin
class HashTable<K, V>(private val capacity: Int = 16) {
    private data class Entry<K, V>(val key: K, var value: V)
    
    private val table = Array<MutableList<Entry<K, V>>?>(capacity) { null }
    private var size = 0
    
    private fun hash(key: K): Int {
        return key.hashCode() and (capacity - 1)
    }
    
    fun put(key: K, value: V) {
        val index = hash(key)
        
        if (table[index] == null) {
            table[index] = mutableListOf()
        }
        
        val bucket = table[index]!!
        
        // Check if key already exists
        for (entry in bucket) {
            if (entry.key == key) {
                entry.value = value
                return
            }
        }
        
        // Add new entry
        bucket.add(Entry(key, value))
        size++
    }
    
    fun get(key: K): V? {
        val index = hash(key)
        val bucket = table[index] ?: return null
        
        for (entry in bucket) {
            if (entry.key == key) {
                return entry.value
            }
        }
        return null
    }
    
    fun remove(key: K): V? {
        val index = hash(key)
        val bucket = table[index] ?: return null
        
        val iterator = bucket.iterator()
        while (iterator.hasNext()) {
            val entry = iterator.next()
            if (entry.key == key) {
                iterator.remove()
                size--
                return entry.value
            }
        }
        return null
    }
    
    fun contains(key: K): Boolean {
        return get(key) != null
    }
    
    fun size() = size
    fun isEmpty() = size == 0
    
    fun display() {
        for (i in table.indices) {
            val bucket = table[i]
            if (bucket != null && bucket.isNotEmpty()) {
                println("Index $i: ${bucket.map { "${it.key}=${it.value}" }}")
            }
        }
    }
}
```

#### Time Complexity
- **Average Case**: O(1) for all operations
- **Worst Case**: O(n) for all operations (when all keys hash to same bucket)

#### Use Cases
- Database indexing
- Caching
- Symbol tables in compilers
- Associative arrays

---

## Advanced Data Structures

### Heaps

A heap is a complete binary tree that satisfies the heap property.

```
Visual (Max Heap):
        100
       /   \
      80    60
     / \   /
    40 20 10
```

#### Max Heap Implementation
```kotlin
class MaxHeap<T : Comparable<T>> {
    private val heap = mutableListOf<T>()
    
    fun insert(element: T) {
        heap.add(element)
        heapifyUp(heap.size - 1)
    }
    
    fun extractMax(): T? {
        if (heap.isEmpty()) return null
        
        val max = heap[0]
        heap[0] = heap[heap.size - 1]
        heap.removeAt(heap.size - 1)
        
        if (heap.isNotEmpty()) {
            heapifyDown(0)
        }
        
        return max
    }
    
    fun peek(): T? = if (heap.isEmpty()) null else heap[0]
    
    private fun heapifyUp(index: Int) {
        val parentIndex = (index - 1) / 2
        
        if (parentIndex >= 0 && heap[index] > heap[parentIndex]) {
            swap(index, parentIndex)
            heapifyUp(parentIndex)
        }
    }
    
    private fun heapifyDown(index: Int) {
        val leftChild = 2 * index + 1
        val rightChild = 2 * index + 2
        var largest = index
        
        if (leftChild < heap.size && heap[leftChild] > heap[largest]) {
            largest = leftChild
        }
        
        if (rightChild < heap.size && heap[rightChild] > heap[largest]) {
            largest = rightChild
        }
        
        if (largest != index) {
            swap(index, largest)
            heapifyDown(largest)
        }
    }
    
    private fun swap(i: Int, j: Int) {
        val temp = heap[i]
        heap[i] = heap[j]
        heap[j] = temp
    }
    
    fun size() = heap.size
    fun isEmpty() = heap.isEmpty()
    
    fun display() {
        println("Heap: $heap")
    }
}

// Priority Queue using Heap
class PriorityQueue<T : Comparable<T>> {
    private val heap = MaxHeap<T>()
    
    fun enqueue(element: T) = heap.insert(element)
    fun dequeue(): T? = heap.extractMax()
    fun peek(): T? = heap.peek()
    fun isEmpty() = heap.isEmpty()
    fun size() = heap.size()
}
```

#### Time Complexity
- **Insert**: O(log n)
- **Extract Max**: O(log n)
- **Peek**: O(1)

#### Use Cases
- Priority queues
- Heap sort algorithm
- Finding k largest/smallest elements
- Graph algorithms (Dijkstra, Prim's)

---

### Tries

A trie is a tree data structure used for efficient string searching.

```
Visual:
        root
       /    \
      c      t
     /        \
    a          h
   /            \
  t              e
 /               
""                
```

#### Implementation
```kotlin
class TrieNode {
    val children = mutableMapOf<Char, TrieNode>()
    var isEndOfWord = false
}

class Trie {
    private val root = TrieNode()
    
    fun insert(word: String) {
        var current = root
        
        for (char in word) {
            if (char !in current.children) {
                current.children[char] = TrieNode()
            }
            current = current.children[char]!!
        }
        
        current.isEndOfWord = true
    }
    
    fun search(word: String): Boolean {
        var current = root
        
        for (char in word) {
            if (char !in current.children) {
                return false
            }
            current = current.children[char]!!
        }
        
        return current.isEndOfWord
    }
    
    fun startsWith(prefix: String): Boolean {
        var current = root
        
        for (char in prefix) {
            if (char !in current.children) {
                return false
            }
            current = current.children[char]!!
        }
        
        return true
    }
    
    fun getAllWordsWithPrefix(prefix: String): List<String> {
        val result = mutableListOf<String>()
        var current = root
        
        // Navigate to prefix
        for (char in prefix) {
            if (char !in current.children) {
                return emptyList()
            }
            current = current.children[char]!!
        }
        
        // DFS to find all words
        fun dfs(node: TrieNode, currentWord: String) {
            if (node.isEndOfWord) {
                result.add(currentWord)
            }
            
            for ((char, childNode) in node.children) {
                dfs(childNode, currentWord + char)
            }
        }
        
        dfs(current, prefix)
        return result
    }
    
    fun delete(word: String) {
        fun deleteHelper(node: TrieNode, word: String, index: Int): Boolean {
            if (index == word.length) {
                if (!node.isEndOfWord) return false
                node.isEndOfWord = false
                return node.children.isEmpty()
            }
            
            val char = word[index]
            val childNode = node.children[char] ?: return false
            
            val shouldDeleteChild = deleteHelper(childNode, word, index + 1)
            
            if (shouldDeleteChild) {
                node.children.remove(char)
                return !node.isEndOfWord && node.children.isEmpty()
            }
            
            return false
        }
        
        deleteHelper(root, word, 0)
    }
}
```

#### Time Complexity
- **Insert**: O(m) where m is length of word
- **Search**: O(m)
- **Delete**: O(m)

#### Use Cases
- Auto-completion
- Spell checkers
- IP routing
- Dictionary implementations

---

## Time & Space Complexity

### Big O Notation Quick Reference

| Operation | Array | Linked List | Stack | Queue | BST (avg) | Hash Table (avg) | Heap |
|-----------|-------|-------------|-------|-------|-----------|------------------|------|
| Access    | O(1)  | O(n)        | O(n)  | O(n)  | O(log n)  | O(1)            | O(n) |
| Search    | O(n)  | O(n)        | O(n)  | O(n)  | O(log n)  | O(1)            | O(n) |
| Insert    | O(n)  | O(1)        | O(1)  | O(1)  | O(log n)  | O(1)            | O(log n) |
| Delete    | O(n)  | O(1)        | O(1)  | O(1)  | O(log n)  | O(1)            | O(log n) |

### Space Complexity
- **Arrays**: O(n)
- **Linked Lists**: O(n)
- **Trees**: O(n)
- **Graphs**: O(V + E)
- **Hash Tables**: O(n)

---

## Practice Problems

### 1. Two Sum (Hash Table)
```kotlin
fun twoSum(nums: IntArray, target: Int): IntArray {
    val map = mutableMapOf<Int, Int>()
    
    for (i in nums.indices) {
        val complement = target - nums[i]
        if (map.containsKey(complement)) {
            return intArrayOf(map[complement]!!, i)
        }
        map[nums[i]] = i
    }
    
    return intArrayOf()
}
```

### 2. Valid Parentheses (Stack)
```kotlin
fun isValid(s: String): Boolean {
    val stack = Stack<Char>()
    val pairs = mapOf(')' to '(', '}' to '{', ']' to '[')
    
    for (char in s) {
        when (char) {
            '(', '{', '[' -> stack.push(char)
            ')', '}', ']' -> {
                if (stack.isEmpty() || stack.pop() != pairs[char]) {
                    return false
                }
            }
        }
    }
    
    return stack.isEmpty()
}
```

### 3. Level Order Traversal (Queue)
```kotlin
fun levelOrder(root: TreeNode<Int>?): List<List<Int>> {
    if (root == null) return emptyList()
    
    val result = mutableListOf<List<Int>>()
    val queue = Queue<TreeNode<Int>>()
    queue.enqueue(root)
    
    while (!queue.isEmpty()) {
        val levelSize = queue.size()
        val currentLevel = mutableListOf<Int>()
        
        repeat(levelSize) {
            val node = queue.dequeue()!!
            currentLevel.add(node.data)
            
            node.left?.let { queue.enqueue(it) }
            node.right?.let { queue.enqueue(it) }
        }
        
        result.add(currentLevel)
    }
    
    return result
}
```

### 4. Reverse Linked List
```kotlin
fun reverseList(head: Node<Int>?): Node<Int>? {
    var prev: Node<Int>? = null
    var current = head
    
    while (current != null) {
        val nextTemp = current.next
        current.next = prev
        prev = current
        current = nextTemp
    }
    
    return prev
}
```

### 5. Find K Largest Elements (Heap)
```kotlin
fun findKLargest(nums: IntArray, k: Int): IntArray {
    val minHeap = java.util.PriorityQueue<Int>()
    
    for (num in nums) {
        minHeap.offer(num)
        if (minHeap.size > k) {
            minHeap.poll()
        }
    }
    
    return minHeap.toIntArray()
}
```

### 6. Implement LRU Cache
```kotlin
class LRUCache(private val capacity: Int) {
    private data class Node(
        val key: Int,
        var value: Int,
        var prev: Node? = null,
        var next: Node? = null
    )
    
    private val cache = mutableMapOf<Int, Node>()
    private val head = Node(0, 0) // Dummy head
    private val tail = Node(0, 0) // Dummy tail
    
    init {
        head.next = tail
        tail.prev = head
    }
    
    fun get(key: Int): Int {
        val node = cache[key] ?: return -1
        moveToHead(node)
        return node.value
    }
    
    fun put(key: Int, value: Int) {
        val existingNode = cache[key]
        
        if (existingNode != null) {
            existingNode.value = value
            moveToHead(existingNode)
        } else {
            val newNode = Node(key, value)
            
            if (cache.size >= capacity) {
                val tail = removeTail()
                cache.remove(tail.key)
            }
            
            cache[key] = newNode
            addToHead(newNode)
        }
    }
    
    private fun addToHead(node: Node) {
        node.prev = head
        node.next = head.next
        head.next?.prev = node
        head.next = node
    }
    
    private fun removeNode(node: Node) {
        node.prev?.next = node.next
        node.next?.prev = node.prev
    }
    
    private fun moveToHead(node: Node) {
        removeNode(node)
        addToHead(node)
    }
    
    private fun removeTail(): Node {
        val lastNode = tail.prev!!
        removeNode(lastNode)
        return lastNode
    }
}
```

### 7. Graph: Detect Cycle in Undirected Graph
```kotlin
fun hasCycle(graph: Graph<Int>, vertices: Int): Boolean {
    val visited = BooleanArray(vertices)
    
    fun dfs(vertex: Int, parent: Int): Boolean {
        visited[vertex] = true
        
        for (neighbor in graph.getNeighbors(vertex)) {
            if (!visited[neighbor]) {
                if (dfs(neighbor, vertex)) return true
            } else if (neighbor != parent) {
                return true
            }
        }
        
        return false
    }
    
    for (i in 0 until vertices) {
        if (!visited[i]) {
            if (dfs(i, -1)) return true
        }
    }
    
    return false
}
```

### 8. Trie: Word Search II
```kotlin
fun findWords(board: Array<CharArray>, words: Array<String>): List<String> {
    val trie = Trie()
    val result = mutableListOf<String>()
    
    // Build trie
    for (word in words) {
        trie.insert(word)
    }
    
    val rows = board.size
    val cols = board[0].size
    val visited = Array(rows) { BooleanArray(cols) }
    
    fun dfs(row: Int, col: Int, node: TrieNode, path: String) {
        if (row < 0 || row >= rows || col < 0 || col >= cols || visited[row][col]) {
            return
        }
        
        val char = board[row][col]
        val nextNode = node.children[char] ?: return
        
        val newPath = path + char
        if (nextNode.isEndOfWord && newPath !in result) {
            result.add(newPath)
        }
        
        visited[row][col] = true
        
        // Explore all 4 directions
        dfs(row + 1, col, nextNode, newPath)
        dfs(row - 1, col, nextNode, newPath)
        dfs(row, col + 1, nextNode, newPath)
        dfs(row, col - 1, nextNode, newPath)
        
        visited[row][col] = false
    }
    
    for (i in 0 until rows) {
        for (j in 0 until cols) {
            dfs(i, j, trie.root, "")
        }
    }
    
    return result
}
```

---

## Advanced Topics

### 1. Segment Trees
```kotlin
class SegmentTree(private val arr: IntArray) {
    private val tree = IntArray(4 * arr.size)
    
    init {
        build(1, 0, arr.size - 1)
    }
    
    private fun build(node: Int, start: Int, end: Int) {
        if (start == end) {
            tree[node] = arr[start]
        } else {
            val mid = (start + end) / 2
            build(2 * node, start, mid)
            build(2 * node + 1, mid + 1, end)
            tree[node] = tree[2 * node] + tree[2 * node + 1]
        }
    }
    
    fun update(idx: Int, value: Int) {
        update(1, 0, arr.size - 1, idx, value)
    }
    
    private fun update(node: Int, start: Int, end: Int, idx: Int, value: Int) {
        if (start == end) {
            arr[idx] = value
            tree[node] = value
        } else {
            val mid = (start + end) / 2
            if (idx <= mid) {
                update(2 * node, start, mid, idx, value)
            } else {
                update(2 * node + 1, mid + 1, end, idx, value)
            }
            tree[node] = tree[2 * node] + tree[2 * node + 1]
        }
    }
    
    fun query(left: Int, right: Int): Int {
        return query(1, 0, arr.size - 1, left, right)
    }
    
    private fun query(node: Int, start: Int, end: Int, left: Int, right: Int): Int {
        if (right < start || end < left) return 0
        if (left <= start && end <= right) return tree[node]
        
        val mid = (start + end) / 2
        val p1 = query(2 * node, start, mid, left, right)
        val p2 = query(2 * node + 1, mid + 1, end, left, right)
        return p1 + p2
    }
}
```

### 2. Union-Find (Disjoint Set)
```kotlin
class UnionFind(n: Int) {
    private val parent = IntArray(n) { it }
    private val rank = IntArray(n) { 0 }
    
    fun find(x: Int): Int {
        if (parent[x] != x) {
            parent[x] = find(parent[x]) // Path compression
        }
        return parent[x]
    }
    
    fun union(x: Int, y: Int) {
        val rootX = find(x)
        val rootY = find(y)
        
        if (rootX != rootY) {
            // Union by rank
            when {
                rank[rootX] < rank[rootY] -> parent[rootX] = rootY
                rank[rootX] > rank[rootY] -> parent[rootY] = rootX
                else -> {
                    parent[rootY] = rootX
                    rank[rootX]++
                }
            }
        }
    }
    
    fun connected(x: Int, y: Int): Boolean {
        return find(x) == find(y)
    }
}
```

### 3. Fenwick Tree (Binary Indexed Tree)
```kotlin
class FenwickTree(private val n: Int) {
    private val tree = IntArray(n + 1)
    
    fun update(idx: Int, delta: Int) {
        var i = idx + 1
        while (i <= n) {
            tree[i] += delta
            i += i and (-i)
        }
    }
    
    fun query(idx: Int): Int {
        var sum = 0
        var i = idx + 1
        while (i > 0) {
            sum += tree[i]
            i -= i and (-i)
        }
        return sum
    }
    
    fun rangeQuery(left: Int, right: Int): Int {
        return query(right) - if (left > 0) query(left - 1) else 0
    }
}
```

---

## Testing Your Implementation

### Unit Tests Example
```kotlin
fun testDataStructures() {
    // Test Stack
    val stack = Stack<Int>()
    stack.push(1)
    stack.push(2)
    stack.push(3)
    assert(stack.pop() == 3)
    assert(stack.peek() == 2)
    
    // Test Queue
    val queue = Queue<String>()
    queue.enqueue("first")
    queue.enqueue("second")
    assert(queue.dequeue() == "first")
    assert(queue.front() == "second")
    
    // Test BST
    val bst = BST<Int>()
    bst.insert(5)
    bst.insert(3)
    bst.insert(7)
    assert(bst.search(3) == true)
    assert(bst.search(4) == false)
    
    // Test Hash Table
    val hashTable = HashTable<String, Int>()
    hashTable.put("apple", 5)
    hashTable.put("banana", 3)
    assert(hashTable.get("apple") == 5)
    assert(hashTable.contains("banana") == true)
    
    println("All tests passed! ✅")
}
```

---

## Performance Comparison

### When to Use Each Data Structure

| Use Case | Best Data Structure | Reason |
|----------|-------------------|---------|
| Fast random access | Array | O(1) access by index |
| Frequent insertions/deletions | Linked List | O(1) at head/tail |
| LIFO operations | Stack | Perfect for undo, recursion |
| FIFO operations | Queue | Perfect for scheduling |
| Fast search with ordering | BST | O(log n) search |
| Fast key-value lookup | Hash Table | O(1) average access |
| Priority-based operations | Heap | O(log n) priority ops |
| String prefix matching | Trie | O(m) where m = string length |
| Graph traversal | Graph | Specialized for connections |

---

## Memory Management Tips

### Kotlin-Specific Optimizations
```kotlin
// Use data classes for nodes to get free equals/hashCode
data class OptimizedNode<T>(val data: T, var next: OptimizedNode<T>? = null)

// Use inline classes for type safety without overhead
@JvmInline
value class NodeId(val value: Int)

// Use collections efficiently
val efficientList = ArrayList<Int>(1000) // Pre-allocate capacity
val efficientSet = HashSet<String>(16, 0.75f) // Custom load factor

// Memory-efficient implementations
class CompactIntArray(size: Int) {
    private val data = IntArray(size)
    var size = 0
        private set
    
    fun add(value: Int) {
        if (size < data.size) {
            data[size++] = value
        }
    }
    
    operator fun get(index: Int): Int = data[index]
}
```

---

## Summary

This comprehensive guide covers the fundamental data structures every programmer should know:

### Key Takeaways:
1. **Choose the right tool**: Each data structure has specific strengths
2. **Understand trade-offs**: Time vs space complexity
3. **Practice implementation**: Build from scratch to understand internals
4. **Apply to real problems**: Connect theory to practical applications

### Next Steps:
1. Implement each data structure from scratch
2. Solve the practice problems
3. Explore advanced topics like self-balancing trees
4. Study algorithm patterns that use these structures

### Resources for Further Learning:
- **Books**: "Introduction to Algorithms" by Cormen et al.
- **Online**: LeetCode, HackerRank for practice problems
- **Visualization**: VisuAlgo.net for animated explanations

Remember: The best way to learn data structures is by implementing them yourself and solving real problems. Start with the basics and gradually move to more complex structures as you build confidence.

Happy coding! 🚀
