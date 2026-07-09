# vhs tape -> mpsify-demo.gif
# render:  vhs demo.tape
Output mpsify-demo.gif

Set Shell "bash"
Set FontSize 18
Set Width 1100
Set Height 470
Set Padding 20
Set Theme "Dracula"
Set TypingSpeed 45ms

Type 'source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate MPS && cd ~/Desktop/my/project/convMPS'
Enter
Sleep 1.5s
Ctrl+L
Sleep 500ms

Type "# A repo that hardcodes .cuda() — dies on a Mac"
Enter
Sleep 800ms
Type "python examples/demo.py"
Sleep 500ms
Enter
Sleep 2.5s

Type "# Same script, zero edits, under mpsify:"
Enter
Sleep 800ms
Type "python -m mpsify examples/demo.py"
Sleep 500ms
Enter
Sleep 4s

Type "# trained on mps:0. that's the whole tool."
Enter
Sleep 2.5s
