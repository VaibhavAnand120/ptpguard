SCENARIOS = {
    "genuine_feasible": [
        ("agent", "Sir kab payment kar paoge?"),
        ("borrower", "10 tareekh ko salary aayegi."),
        ("borrower", "Main 10 tareekh ko 2000 pay kar dunga."),
        ("agent", "2000 on 10th confirmed?"),
        ("borrower", "Haan sir, confirm hai.")
    ],
    "genuine_hardship": [
        ("agent", "Sir aap payment kab kar paoge?"),
        ("borrower", "Sir abhi paise nahi hain."),
        ("borrower", "Salary 15 ko aayegi, uske baad kar dunga."),
        ("agent", "15 ke baad payment kar paoge?"),
        ("borrower", "Haan sir, salary ke baad kar dunga.")
    ],
    "escape": [
        ("agent", "Sir pending amount kab pay karenge?"),
        ("borrower", "Haan haan sir kar dunga."),
        ("agent", "Kab karenge?"),
        ("borrower", "Bas call rakhiye sir, kar dunga.")
    ],
    "third_party": [
        ("agent", "Sir payment kab karenge?"),
        ("third_party", "Main unka bhai bol raha hoon, main payment kar dunga.")
    ],
    "agent_pushed": [
        ("agent", "Sir ₹5000 10th ko kar denge, correct?"),
        ("borrower", "Ji."),
        ("agent", "Okay, I'll mark ₹5000 on 10th.")
    ],
    "background_family": [
        ("agent", "Namaste Sir, ABC Finance se bol raha hoon. Aapka ₹4000 overdue chal raha hai, kab clear karenge?"),
        ("borrower", "Haan sir, main dekh raha tha..."),
        ("third_party_background", "(Peeche se patni: Arey bolo paise nahi hai abhi, ration lena hai, phone kaato unka!)"),
        ("borrower", "Sir abhi paise nahi hain, salary late hai."),
        ("agent", "Sir kya 12 tareekh ko payment possible hai?"),
        ("third_party_background", "(Wife whispering in background: 4000 mat bolo, bolo 1000 se zyada nahi ho payega)"),
        ("borrower", "Sir 12 tareekh ko ₹1000 hi pay kar paunga.")
    ],
    "auto_diarize_live": [
        ("auto", "Namaste sir, Bajaj Finance se call hai regarding your overdue loan EMI ₹3500."),
        ("auto", "Sir abhi mere paas paise nahi hain, dukaan me nuksan hua hai."),
        ("auto", "(Peeche se patni: Bol do kal baat karenge, phone cut karo!)"),
        ("auto", "Main 15 tareekh ko ₹2000 transfer kar dunga pakka.")
    ],
    "reversal": [
        ("agent", "Can you pay on the 10th?"),
        ("borrower", "If salary comes, I will try to pay on the 10th."),
        ("borrower", "Salary has already come, account me paise aa gaye."),
        ("borrower", "I definitely confirm 2000 pay kar dunga on 10th.")
    ]
}
