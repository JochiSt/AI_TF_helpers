def merge_dicts(start, add):
    z = start.copy()   # start with keys and values of x
    z.update(add)    # modifies z with keys and values of y
    return z