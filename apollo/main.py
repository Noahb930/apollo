import os
from json import load, dump, loads, dumps
import logging
import itertools
from cmd import Cmd
import base64
from io import BytesIO
from PIL import Image
import sys
from cefpython3 import cefpython as cef

from agent import Agent

def main():
    sys.excepthook = cef.ExceptHook
    cef.Initialize()
    browser = cef.CreateBrowserSync(url="file://"+os.path.join(os.path.dirname(os.path.abspath(__file__)),'../html/dashboard.html'),window_title="Hello World!")
    set_javascript_bindings(browser, Agent(),["load","evaluate","predict"])
    cef.MessageLoop()
    cef.Shutdown()

def set_javascript_bindings(browser, agent, commands):
    bindings = cef.JavascriptBindings(bindToFrames=False, bindToPopups=False)
    for command in commands:
        bindings.SetFunction(command, generate_async_message(agent, command))
    browser.SetJavascriptBindings(bindings)

def generate_async_message(agent, command):
    def async_message(params, callback=None):
        returned = getattr(agent, command)(*params)
        if callback:
            callback.Call(*returned)
        else:
            return returned
    return async_message

if __name__ == '__main__':
    main()
