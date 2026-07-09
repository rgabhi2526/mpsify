# vhs demo-short.tape  ->  mpsify-demo-short.gif  (LinkedIn, <10s)
Output mpsify-demo-short.gif

Set Shell "bash"
Set FontSize 18
Set Width 1100
Set Height 460
Set Padding 20
Set Theme "Dracula"
Set TypingSpeed 22ms

Hide
Type 'source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate MPS && cd ~/Desktop/my/project/convMPS && clear'
Enter
Sleep 2s
Show

Type "# CUDA-hardcoded script — dies on a Mac"
Enter
Type "python examples/demo.py"
Enter
Sleep 2.3s

Type "# zero edits, under mpsify:"
Enter
Type "python -m mpsify examples/demo.py"
Enter
Sleep 4.7s
