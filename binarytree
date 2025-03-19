#include <stdio.h>
#include <stdlib.h>

typedef struct Node {
    int data;
    struct Node* left;
    struct Node* right;
} Node;

Node* createNode(int value) {
    Node* newNode = (Node*)malloc(sizeof(Node));
    newNode->data = value;
    newNode->left = newNode->right = NULL;
    return newNode;
}

Node* addElement(Node* root, int value) {
    if (root == NULL) {
        return createNode(value);
    }

    if (value < root->data) {
        root->left = addElement(root->left, value);
    } else {
        root->right = addElement(root->right, value);
    }

    return root;
}

void printTree(Node* root) {
    if (root != NULL) {
        printTree(root->left);
        printf("%d ", root->data);
        printTree(root->right);
    }
}
Node* BinarySearch(int data, Node* node) {
    Node* stack[10] = {NULL};
    stack[0] = node;
    int counter = 0;

    while (stack[0] != NULL) {
        if (stack[0]->data == data) {
            return stack[0];
        }

        if (stack[0]->left != NULL) {
            int iterator = 0;
            while (stack[iterator] != NULL) {
                iterator++;
            }
            stack[iterator] = stack[0]->left;
        }

        if (stack[0]->right != NULL) {
            int iterator = 0;
            while (stack[iterator] != NULL) {
                iterator++;
            }
            stack[iterator] = stack[0]->right;
        }

        int iterator = 0;
        while (stack[iterator + 1] != NULL) {
            stack[iterator] = stack[iterator + 1];
            iterator++;
        }
        stack[iterator] = NULL; 

        counter++;
    }
    return NULL;
}
void StackCreator(int* stack, Node* root) {
    if (root->left != NULL) {
        int iterator = 0;
        while (stack[iterator] != -1) {
            iterator++;
        }
        stack[iterator] = root->left->data;
        StackCreator(stack, root->left);
    }

    if (root->right != NULL) {
        int iterator = 0;
        while (stack[iterator] != -1) {
            iterator++;
        }
        stack[iterator] = root->right->data;
        StackCreator(stack, root->right);
    }
}

void DeleteElement(int data, Node* root) {
    Node* FindParent(Node* node,Node* root){
        if(root == NULL){
            return NULL;
        }
        if(root->left == node){
            return root;
        }
        
        if(root->right == node){
            return root;
        }
        FindParent(node,root->left);
        FindParent(node,root->right);
    }
    Node* node = BinarySearch(data, root);

    if (node == NULL) {
        return;
    }

    int stack[10] = {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1};

    StackCreator(stack, node);
    Node* parent = FindParent(node,root);
    if (parent->left == node){
        parent->left = NULL;
        
    }else{
        parent->right = NULL;
    }
    for (int i = 0; i < 10; i++) {
        if (stack[i] != -1) {
            root = addElement(root,stack[i]);
        }
    }
}

int main() {
    Node* root = NULL;

    root = addElement(root, 5);
    root = addElement(root, 3);
    root = addElement(root, 7);
    root = addElement(root, 2);
    root = addElement(root, 4);
    root = addElement(root, 6);
    root = addElement(root, 8);

    printTree(root);
    DeleteElement(3,root);

    printTree(root);

    return 0;
}
