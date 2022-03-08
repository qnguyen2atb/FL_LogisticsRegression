#!/bin/sh

sudo ps aux | grep Netskope | grep -v grep | awk '{ print "kill -9", $2 }' | sudo sh
echo '[✓] Kill Netskope Process'

sudo rm -rf /Applications/Remove\ Netskope\ Client.app
echo '[✓] Removed Remove Netskope Client.app'

sudo rm -rf /Library/Application\ Support/Netskope
echo '[✓] Removed Agent of Netskope Client.app'

echo 'Successfully uninstalled.'