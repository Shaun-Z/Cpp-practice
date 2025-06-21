#include <iostream>

int main() {
    using namespace std;
    
    int carrots;

    carrots = 25;

    cout << "I have ";
    cout << carrots;
    cout << " carrots." << endl; // Output the number of carrots
    carrots = carrots - 1; // Decrease the number of carrots by 1
    cout << "Crunch, crunch. Now I have ";
    cout << carrots;
    cout << " carrots." << endl; // Output the updated number of carrots
    return 0; // Return 0 to indicate successful completion
}