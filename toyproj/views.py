__author__ = 'fanxn'

from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from datetime import datetime
from django.shortcuts import redirect
import os, json


def index(request):
    if request.method == "GET":
        return render(request, "index.html")
