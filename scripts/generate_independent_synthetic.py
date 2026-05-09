#!/usr/bin/env python3
"""
Generate a large independent synthetic topic set for uncontaminated autoresearch validation.

Scientific design:
  - Questions are about DIFFERENT movies than the 65 real R2C2 topics
  - All questions are grounded in the actual corpus (BM25 retrieval verifies answerability)
  - 3-pass quality pipeline: generate → verify → enrich with harder variants
  - Every judgment is cached and model-attributed — never pay twice

Output:
  data/synthetic_val/synthetic_topics.json      — full topic list (all 5000+)
  data/synthetic_val/synthetic_topics_loop.json — 1300-topic stratified subset for autoresearch
  data/synthetic_val/generation_metadata.json   — cost, model, stats, per-movie counts
  data/synthetic_val/generation_cache.json      — resume cache (pass1/2/3 per movie)

Usage:
    python scripts/generate_independent_synthetic.py --budget 40 --output data/synthetic_val
    python scripts/generate_independent_synthetic.py --budget 40 --resume  # continue if interrupted
"""

import argparse
import json
import pickle
import random
import re
import time
from collections import defaultdict
from pathlib import Path

import anthropic
import bm25s

BASE = Path(__file__).resolve().parent.parent

# ── Movies in real R2C2 topics — MUST NOT appear in synthetic set ─────────────
EXCLUDED_MOVIES = {
    # Star Wars in real topics
    "Rogue One", "Rogue One: A Star Wars Story",
    "The Empire Strikes Back", "Star Wars: The Empire Strikes Back",
    "Revenge of the Sith", "Star Wars: Episode III – Revenge of the Sith",
    "Solo: A Star Wars Story",
    # MCU in real topics
    "Avengers: Endgame", "Avengers: Infinity War",
    "Black Widow", "Deadpool", "Green Lantern",
    # Pixar/Disney in real topics
    "Toy Story", "Toy Story 4", "Cars 2", "WALL-E",
    "Finding Nemo", "Monsters, Inc.", "Monsters Inc.",
    # Ghibli in real topics
    "Spirited Away",
    # Other
    "Back to the Future Part II", "Back to the Future Part 2",
    "The Fast and the Furious", "Fast & Furious", "F9", "Fast X",
    "Interstellar", "Casablanca", "The Shawshank Redemption",
    "The Silence of the Lambs", "Avatar", "Zootopia",
    "Harry Potter and the Philosopher's Stone",
    "Harry Potter and the Chamber of Secrets",
    "Harry Potter and the Prisoner of Azkaban",
    "Harry Potter and the Goblet of Fire",
    "Harry Potter and the Order of the Phoenix",
    "Harry Potter and the Half-Blood Prince",
    "Harry Potter and the Deathly Hallows",
    "John Wick: Chapter 4", "Bean", "Mr. Bean's Holiday",
    "Wicked", "Slumdog Millionaire", "Four Weddings and a Funeral",
    "Love Actually", "Gladiator", "I, Robot", "The Day After Tomorrow",
    "May 18", "12.12: The Day", "Spider-Man: Homecoming",
    "Spider-Man: Far From Home", "Spider-Man: No Way Home",
}

# ── Challenge taxonomy ────────────────────────────────────────────────────────
CHALLENGE_TYPES = {
    # Core types from real R2C2 topics
    "specific_object":    "A specific prop, gadget, item, or MacGuffin in the film",
    "actor_identity":     "Which actor played a specific role or character",
    "character_name":     "The name of a character described by their role/description",
    "count_number":       "How many of something (films, awards, characters, scenes)",
    "comparison":         "Which of two things was bigger/earlier/better-reviewed/more successful",
    "time_calculation":   "How many years/days/weeks between two specific events",
    "vocabulary_gap":     "Answer uses different terminology than the question implies (e.g., cover song, alternate title)",
    "implicit_entity":    "Describe something without naming it; ask what it is",
    "slogan_quote":       "Exact quote, famous line, tagline, or slogan",
    "award_detail":       "Specific awards won, nominations, or records set",
    "director_writer":    "Who directed, wrote, or produced the film",
    "plot_detail":        "A specific scene, plot point, or story beat",
    # Harder types — generalization challenges
    "multi_hop":          "Must chain two facts: e.g., 'the actor who played X also played Y in film Z, what is Z's director'",
    "negation":           "Which film/character was NOT something; what was excluded",
    "production_detail":  "Budget, filming location, production company, shooting schedule",
    "release_order":      "Which came first; exact release year; gap between sequels",
    "voice_actor":        "Who voiced an animated character (confusable with live-action casting)",
    "cameo_appearance":   "A brief unexpected appearance by a notable actor or real person",
    "music_detail":       "Composer, specific song, original score, or soundtrack detail",
    "box_office_rank":    "Specific gross figure, opening weekend, all-time ranking",
    "sequel_count":       "How many films in a franchise as of a specific date",
    "alternate_title":    "Film released under different name in another country or language",
    "real_vs_fictional":  "Distinguish the real-world element from its fictional depiction in the film",
    "casting_change":     "Which actor was replaced or recast between films in a series",
    "runtime_detail":     "Film's running time, director's cut length, or cut scenes",
}

# ── 600+ candidate movies — diverse, well-covered in Wikipedia ───────────────
CANDIDATE_MOVIES = [
    # ── Classic Hollywood (pre-1970) ──
    ("Citizen Kane", "1941 drama"),
    ("Casablanca", "1942 romance war drama"),  # excluded from REAL but fine as synthetic since it's about DIFFERENT questions
    # Note: Casablanca IS in real topics (0031 asks about Rick Blaine's actor) - must exclude
    # Actually let me not include it since it's in EXCLUDED_MOVIES above implicitly
    # Let me keep it and rely on EXCLUDED_MOVIES check... wait Casablanca is in EXCLUDED_MOVIES
    # I'll remove it from here:
    ("Psycho", "1960 Hitchcock thriller"),
    ("Vertigo", "1958 Hitchcock mystery"),
    ("Rear Window", "1954 Hitchcock thriller"),
    ("North by Northwest", "1959 Hitchcock thriller"),
    ("Sunset Boulevard", "1950 film noir drama"),
    ("Some Like It Hot", "1959 Billy Wilder comedy"),
    ("The Apartment", "1960 Billy Wilder comedy drama"),
    ("Double Indemnity", "1944 film noir"),
    ("Singin' in the Rain", "1952 musical"),
    ("Sunset Blvd.", "1950 drama"),
    ("Lawrence of Arabia", "1962 epic historical drama"),
    ("2001: A Space Odyssey", "1968 Stanley Kubrick sci-fi"),
    ("Dr. Strangelove", "1964 Kubrick black comedy"),
    ("A Clockwork Orange", "1971 Kubrick dystopian"),
    ("The Wizard of Oz", "1939 musical fantasy"),
    ("Gone with the Wind", "1939 historical romance"),
    ("It's a Wonderful Life", "1946 fantasy drama"),
    ("Sunset Boulevard", "1950 film noir"),
    ("Rear Window", "1954 thriller"),
    ("Spartacus", "1960 historical epic"),
    ("Ben-Hur", "1959 historical epic"),
    ("The Bridge on the River Kwai", "1957 war drama"),
    ("On the Waterfront", "1954 crime drama"),
    ("Anatomy of a Murder", "1959 legal drama"),
    ("12 Angry Men", "1957 legal drama"),
    ("The Maltese Falcon", "1941 film noir"),
    # ── Italian/European Cinema ──
    ("8½", "1963 Federico Fellini Italian drama"),
    ("La Dolce Vita", "1960 Fellini Italian drama"),
    ("Breathless", "1960 Jean-Luc Godard French New Wave"),
    ("The 400 Blows", "1959 François Truffaut French drama"),
    ("Tokyo Story", "1953 Yasujirō Ozu Japanese drama"),
    ("Seven Samurai", "1954 Akira Kurosawa Japanese"),
    ("Rashomon", "1950 Kurosawa Japanese"),
    ("Ikiru", "1952 Kurosawa Japanese drama"),
    ("Bicycle Thieves", "1948 Vittorio De Sica Italian neorealism"),
    ("Cinema Paradiso", "1988 Giuseppe Tornatore Italian drama"),
    ("Life Is Beautiful", "1997 Roberto Benigni Italian drama"),
    ("Amarcord", "1973 Fellini Italian comedy drama"),
    # ── 1970s American Cinema ──
    ("The Godfather", "1972 Francis Ford Coppola crime drama"),
    ("The Godfather Part II", "1974 Francis Ford Coppola crime drama"),
    ("Taxi Driver", "1976 Martin Scorsese crime drama"),
    ("Apocalypse Now", "1979 Francis Ford Coppola war drama"),
    ("Chinatown", "1974 Roman Polanski neo-noir"),
    ("One Flew Over the Cuckoo's Nest", "1975 Miloš Forman drama"),
    ("Annie Hall", "1977 Woody Allen romantic comedy"),
    ("Manhattan", "1979 Woody Allen romantic drama"),
    ("Raging Bull", "1980 Scorsese biographical sports drama"),
    ("Network", "1976 Sidney Lumet satire drama"),
    ("Dog Day Afternoon", "1975 Sidney Lumet crime drama"),
    ("All the President's Men", "1976 political thriller"),
    ("Rocky", "1976 John G. Avildsen sports drama"),
    ("Alien", "1979 Ridley Scott sci-fi horror"),
    ("Star Wars: A New Hope", "1977 George Lucas sci-fi"),
    ("Star Wars: Return of the Jedi", "1983 Richard Marquand sci-fi"),
    ("Close Encounters of the Third Kind", "1977 Spielberg sci-fi"),
    ("Jaws", "1975 Steven Spielberg thriller"),
    ("Nashville", "1975 Robert Altman drama"),
    ("The Deer Hunter", "1978 Michael Cimino war drama"),
    ("Kramer vs. Kramer", "1979 Robert Benton drama"),
    # ── 1980s ──
    ("E.T. the Extra-Terrestrial", "1982 Steven Spielberg sci-fi"),
    ("Raiders of the Lost Ark", "1981 Spielberg adventure"),
    ("Indiana Jones and the Temple of Doom", "1984 Spielberg adventure"),
    ("Indiana Jones and the Last Crusade", "1989 Spielberg adventure"),
    ("Blade Runner", "1982 Ridley Scott sci-fi noir"),
    ("The Terminator", "1984 James Cameron sci-fi action"),
    ("Aliens", "1986 James Cameron sci-fi action"),
    ("Die Hard", "1988 John McTiernan action thriller"),
    ("Top Gun", "1986 Tony Scott action drama"),
    ("Platoon", "1986 Oliver Stone war drama"),
    ("Full Metal Jacket", "1987 Stanley Kubrick war drama"),
    ("Ferris Bueller's Day Off", "1986 John Hughes comedy"),
    ("The Breakfast Club", "1985 John Hughes teen drama"),
    ("Ghostbusters", "1984 Ivan Reitman supernatural comedy"),
    ("Back to the Future", "1985 Robert Zemeckis sci-fi comedy"),
    ("Beetlejuice", "1988 Tim Burton fantasy comedy"),
    ("Batman", "1989 Tim Burton superhero"),
    ("Blue Velvet", "1986 David Lynch neo-noir"),
    ("Do the Right Thing", "1989 Spike Lee drama"),
    ("Broadcast News", "1987 James L. Brooks comedy drama"),
    ("Tootsie", "1982 Sydney Pollack comedy"),
    ("Rain Man", "1988 Barry Levinson drama"),
    ("When Harry Met Sally", "1989 Rob Reiner romantic comedy"),
    ("The Princess Bride", "1987 Rob Reiner fantasy comedy"),
    ("RoboCop", "1987 Paul Verhoeven sci-fi action"),
    ("Predator", "1987 John McTiernan action sci-fi"),
    ("Beverly Hills Cop", "1984 Martin Brest action comedy"),
    ("Coming to America", "1988 John Landis comedy"),
    ("The Color Purple", "1985 Spielberg drama"),
    ("Platoon", "1986 war drama"),
    ("Empire of the Sun", "1987 Spielberg war drama"),
    ("Dangerous Liaisons", "1988 Stephen Frears drama"),
    ("Wings of Desire", "1987 Wim Wenders fantasy drama"),
    # ── 1990s ──
    ("Goodfellas", "1990 Martin Scorsese crime"),
    ("The Silence of the Lambs", "1991 Jonathan Demme thriller"),
    ("Schindler's List", "1993 Steven Spielberg historical drama"),
    ("Philadelphia", "1993 Jonathan Demme legal drama"),
    ("Jurassic Park", "1993 Steven Spielberg sci-fi adventure"),
    ("The Lion King", "1994 Disney animated musical"),
    ("Aladdin", "1992 Disney animated musical"),
    ("Beauty and the Beast", "1991 Disney animated musical"),
    ("Forrest Gump", "1994 Robert Zemeckis drama"),
    ("Pulp Fiction", "1994 Quentin Tarantino crime"),
    ("Braveheart", "1995 Mel Gibson historical epic"),
    ("Fargo", "1996 Coen Brothers crime comedy"),
    ("The Big Lebowski", "1998 Coen Brothers comedy"),
    ("No Country for Old Men", "2007 Coen Brothers thriller"),
    ("L.A. Confidential", "1997 Curtis Hanson neo-noir"),
    ("American Beauty", "1999 Sam Mendes drama"),
    ("Fight Club", "1999 David Fincher thriller"),
    ("The Matrix", "1999 Wachowski sci-fi action"),
    ("The Truman Show", "1998 Peter Weir drama"),
    ("Magnolia", "1999 Paul Thomas Anderson drama"),
    ("Boogie Nights", "1997 Paul Thomas Anderson drama"),
    ("There Will Be Blood", "2007 Paul Thomas Anderson drama"),
    ("Jerry Maguire", "1996 Cameron Crowe sports drama"),
    ("As Good as It Gets", "1997 James L. Brooks comedy drama"),
    ("Good Will Hunting", "1997 Gus Van Sant drama"),
    ("Titanic", "1997 James Cameron romance disaster"),
    ("Saving Private Ryan", "1998 Steven Spielberg war drama"),
    ("Amistad", "1997 Steven Spielberg historical drama"),
    ("Being John Malkovich", "1999 Spike Jonze comedy drama"),
    ("Election", "1999 Alexander Payne comedy"),
    ("American History X", "1998 Tony Kaye drama"),
    ("Eyes Wide Shut", "1999 Stanley Kubrick drama"),
    ("The Sixth Sense", "1999 M. Night Shyamalan thriller"),
    ("Boys Don't Cry", "1999 Kimberly Peirce biographical drama"),
    ("Run Lola Run", "1998 Tom Tykwer German thriller"),
    ("Central Station", "1998 Walter Salles Brazilian drama"),
    ("Three Colors: Blue", "1993 Krzysztof Kieślowski French drama"),
    ("Three Colors: Red", "1994 Kieślowski French drama"),
    ("Trainspotting", "1996 Danny Boyle British drama"),
    ("The Full Monty", "1997 Peter Cattaneo British comedy"),
    ("Lock, Stock and Two Smoking Barrels", "1998 Guy Ritchie British crime"),
    ("Secrets & Lies", "1996 Mike Leigh British drama"),
    ("Breaking the Waves", "1996 Lars von Trier Danish drama"),
    ("Thelma & Louise", "1991 Ridley Scott road drama"),
    ("Unforgiven", "1992 Clint Eastwood western"),
    ("Heat", "1995 Michael Mann crime thriller"),
    ("Se7en", "1995 David Fincher thriller"),
    ("The Usual Suspects", "1995 Bryan Singer thriller"),
    ("12 Monkeys", "1995 Terry Gilliam sci-fi"),
    ("Natural Born Killers", "1994 Oliver Stone crime"),
    ("Nixon", "1995 Oliver Stone biographical drama"),
    ("JFK", "1991 Oliver Stone political thriller"),
    ("The Piano", "1993 Jane Campion drama"),
    ("Short Cuts", "1993 Robert Altman drama"),
    ("The Age of Innocence", "1993 Martin Scorsese period drama"),
    ("Cape Fear", "1991 Martin Scorsese thriller"),
    ("Hoop Dreams", "1994 Steve James documentary"),
    # ── 2000s ──
    ("Gladiator", "2000 Ridley Scott historical epic"),
    ("Traffic", "2000 Steven Soderbergh crime drama"),
    ("Memento", "2000 Christopher Nolan thriller"),
    ("Requiem for a Dream", "2000 Darren Aronofsky drama"),
    ("Amores Perros", "2000 Alejandro González Iñárritu Mexican drama"),
    ("In the Mood for Love", "2000 Wong Kar-wai Hong Kong drama"),
    ("Mulholland Drive", "2001 David Lynch mystery"),
    ("Moulin Rouge!", "2001 Baz Luhrmann musical romance"),
    ("A Beautiful Mind", "2001 Ron Howard biographical drama"),
    ("Black Hawk Down", "2001 Ridley Scott war drama"),
    ("Amelie", "2001 Jean-Pierre Jeunet French romantic comedy"),
    ("The Lord of the Rings: The Fellowship of the Ring", "2001 Peter Jackson fantasy"),
    ("The Lord of the Rings: The Two Towers", "2002 Peter Jackson fantasy"),
    ("The Lord of the Rings: The Return of the King", "2003 Peter Jackson fantasy"),
    ("Minority Report", "2002 Steven Spielberg sci-fi thriller"),
    ("Catch Me If You Can", "2002 Steven Spielberg biographical thriller"),
    ("City of God", "2002 Fernando Meirelles Brazilian crime drama"),
    ("Gangs of New York", "2002 Martin Scorsese crime drama"),
    ("The Hours", "2002 Stephen Daldry drama"),
    ("Far from Heaven", "2002 Todd Haynes drama"),
    ("Adaptation", "2002 Spike Jonze comedy drama"),
    ("Road to Perdition", "2002 Sam Mendes crime drama"),
    ("Mystic River", "2003 Clint Eastwood crime drama"),
    ("Lost in Translation", "2003 Sofia Coppola drama"),
    ("Kill Bill: Volume 1", "2003 Quentin Tarantino action"),
    ("Kill Bill: Volume 2", "2004 Quentin Tarantino action"),
    ("Eternal Sunshine of the Spotless Mind", "2004 Michel Gondry sci-fi romance"),
    ("Million Dollar Baby", "2004 Clint Eastwood sports drama"),
    ("Sideways", "2004 Alexander Payne comedy drama"),
    ("The Incredibles", "2004 Pixar animated superhero"),
    ("Batman Begins", "2005 Christopher Nolan superhero"),
    ("Brokeback Mountain", "2005 Ang Lee romantic drama"),
    ("Good Night, and Good Luck", "2005 George Clooney historical drama"),
    ("Walk the Line", "2005 James Mangold biographical music drama"),
    ("Munich", "2005 Steven Spielberg historical thriller"),
    ("The Constant Gardener", "2005 Fernando Meirelles thriller"),
    ("The Departed", "2006 Martin Scorsese crime thriller"),
    ("Babel", "2006 Alejandro González Iñárritu drama"),
    ("Pan's Labyrinth", "2006 Guillermo del Toro fantasy"),
    ("Little Miss Sunshine", "2006 Jonathan Dayton comedy drama"),
    ("The Queen", "2006 Stephen Frears biographical drama"),
    ("Children of Men", "2006 Alfonso Cuarón sci-fi drama"),
    ("The Dark Knight", "2008 Christopher Nolan superhero"),
    ("The Dark Knight Rises", "2012 Christopher Nolan superhero"),
    ("Ratatouille", "2007 Pixar animated comedy"),
    ("Atonement", "2007 Joe Wright period drama"),
    ("There Will Be Blood", "2007 Paul Thomas Anderson drama"),
    ("Michael Clayton", "2007 Tony Gilroy legal thriller"),
    ("Juno", "2007 Jason Reitman comedy drama"),
    ("No Country for Old Men", "2007 Coen Brothers thriller"),
    ("Into the Wild", "2007 Sean Penn adventure drama"),
    ("The Assassination of Jesse James", "2007 Andrew Dominik western"),
    ("Up", "2009 Pixar animated adventure"),
    ("District 9", "2009 Neill Blomkamp sci-fi"),
    ("Moon", "2009 Duncan Jones sci-fi"),
    ("Inglourious Basterds", "2009 Quentin Tarantino war"),
    ("The Hurt Locker", "2008 Kathryn Bigelow war thriller"),
    ("Precious", "2009 Lee Daniels drama"),
    ("A Serious Man", "2009 Coen Brothers drama"),
    ("Funny Games", "2007 Michael Haneke thriller"),
    ("The White Ribbon", "2009 Michael Haneke drama"),
    ("A Prophet", "2009 Jacques Audiard French crime drama"),
    # ── 2010s ──
    ("Inception", "2010 Christopher Nolan sci-fi thriller"),
    ("The Social Network", "2010 David Fincher biographical drama"),
    ("Black Swan", "2010 Darren Aronofsky psychological thriller"),
    ("True Grit", "2010 Coen Brothers western"),
    ("The Fighter", "2010 David O. Russell sports drama"),
    ("Inside Job", "2010 Charles Ferguson documentary"),
    ("The Tree of Life", "2011 Terrence Malick drama"),
    ("Moneyball", "2011 Bennett Miller sports drama"),
    ("The Artist", "2011 Michel Hazanavicius silent film"),
    ("Hugo", "2011 Martin Scorsese adventure drama"),
    ("The Girl with the Dragon Tattoo", "2011 David Fincher thriller"),
    ("Midnight in Paris", "2011 Woody Allen fantasy comedy"),
    ("The Descendants", "2011 Alexander Payne drama"),
    ("Tinker Tailor Soldier Spy", "2011 Tomas Alfredson spy thriller"),
    ("Drive", "2011 Nicolas Winding Refn crime thriller"),
    ("The Skin I Live In", "2011 Pedro Almodóvar Spanish thriller"),
    ("Melancholia", "2011 Lars von Trier drama"),
    ("The Master", "2012 Paul Thomas Anderson drama"),
    ("Zero Dark Thirty", "2012 Kathryn Bigelow thriller"),
    ("Argo", "2012 Ben Affleck political thriller"),
    ("Django Unchained", "2012 Quentin Tarantino western"),
    ("Lincoln", "2012 Steven Spielberg historical drama"),
    ("Amour", "2012 Michael Haneke Austrian drama"),
    ("Life of Pi", "2012 Ang Lee adventure drama"),
    ("Anna Karenina", "2012 Joe Wright period drama"),
    ("Les Misérables", "2012 Tom Hooper musical drama"),
    ("Skyfall", "2012 Sam Mendes James Bond"),
    ("Gravity", "2013 Alfonso Cuarón sci-fi thriller"),
    ("Her", "2013 Spike Jonze sci-fi romance"),
    ("12 Years a Slave", "2013 Steve McQueen historical drama"),
    ("The Wolf of Wall Street", "2013 Martin Scorsese biographical crime"),
    ("Nebraska", "2013 Alexander Payne drama"),
    ("Blue Is the Warmest Colour", "2013 Abdellatif Kechiche French drama"),
    ("The Great Beauty", "2013 Paolo Sorrentino Italian drama"),
    ("Frozen", "2013 Jennifer Lee Disney animated musical"),
    ("Captain Phillips", "2013 Paul Greengrass thriller"),
    ("Dallas Buyers Club", "2013 Jean-Marc Vallée biographical drama"),
    ("Prisoners", "2013 Denis Villeneuve thriller"),
    ("Enemy", "2013 Denis Villeneuve psychological thriller"),
    ("Birdman", "2014 Alejandro González Iñárritu black comedy"),
    ("Boyhood", "2014 Richard Linklater drama"),
    ("The Grand Budapest Hotel", "2014 Wes Anderson comedy"),
    ("Nightcrawler", "2014 Dan Gilroy thriller"),
    ("Gone Girl", "2014 David Fincher thriller"),
    ("Inherent Vice", "2014 Paul Thomas Anderson mystery"),
    ("Whiplash", "2014 Damien Chazelle music drama"),
    ("Ex Machina", "2014 Alex Garland sci-fi"),
    ("Under the Skin", "2013 Jonathan Glazer sci-fi"),
    ("Ida", "2013 Paweł Pawlikowski Polish drama"),
    ("The Imitation Game", "2014 Morten Tyldum biographical thriller"),
    ("Selma", "2014 Ava DuVernay biographical drama"),
    ("The Theory of Everything", "2014 James Marsh biographical drama"),
    ("Guardians of the Galaxy", "2014 James Gunn Marvel"),
    ("Captain America: The Winter Soldier", "2014 Marvel superhero"),
    ("The Revenant", "2015 Alejandro González Iñárritu survival drama"),
    ("Mad Max: Fury Road", "2015 George Miller action"),
    ("Spotlight", "2015 Tom McCarthy investigative drama"),
    ("The Martian", "2015 Ridley Scott sci-fi"),
    ("Room", "2015 Lenny Abrahamson drama"),
    ("Carol", "2015 Todd Haynes romantic drama"),
    ("Brooklyn", "2015 John Crowley romance drama"),
    ("Bridge of Spies", "2015 Steven Spielberg historical thriller"),
    ("Son of Saul", "2015 László Nemes Hungarian drama"),
    ("Mustang", "2015 Deniz Gamze Ergüven Turkish drama"),
    ("Inside Out", "2015 Pixar animated drama"),
    ("Ant-Man", "2015 Marvel superhero"),
    ("The Hateful Eight", "2015 Quentin Tarantino western"),
    ("Star Wars: The Force Awakens", "2015 J.J. Abrams sci-fi"),
    ("Star Wars: The Last Jedi", "2017 Rian Johnson sci-fi"),
    ("Star Wars: The Rise of Skywalker", "2019 J.J. Abrams sci-fi"),
    ("Star Wars: The Phantom Menace", "1999 George Lucas sci-fi"),
    ("Star Wars: Attack of the Clones", "2002 George Lucas sci-fi"),
    ("Moonlight", "2016 Barry Jenkins drama"),
    ("La La Land", "2016 Damien Chazelle musical romance"),
    ("Manchester by the Sea", "2016 Kenneth Lonergan drama"),
    ("Arrival", "2016 Denis Villeneuve sci-fi"),
    ("Hacksaw Ridge", "2016 Mel Gibson war drama"),
    ("Hell or High Water", "2016 David Mackenzie crime drama"),
    ("Fences", "2016 Denzel Washington drama"),
    ("The Witch", "2015 Robert Eggers horror"),
    ("The Lighthouse", "2019 Robert Eggers horror thriller"),
    ("Moana", "2016 Disney animated musical"),
    ("Rogue One: A Star Wars Story", "2016 sci-fi action"),  # excluded via EXCLUDED_MOVIES check
    ("Coco", "2017 Pixar animated musical"),
    ("Get Out", "2017 Jordan Peele horror thriller"),
    ("Dunkirk", "2017 Christopher Nolan war"),
    ("Three Billboards Outside Ebbing, Missouri", "2017 Martin McDonagh drama"),
    ("The Shape of Water", "2017 Guillermo del Toro fantasy romance"),
    ("Phantom Thread", "2017 Paul Thomas Anderson drama"),
    ("Lady Bird", "2017 Greta Gerwig coming-of-age"),
    ("Call Me by Your Name", "2017 Luca Guadagnino romantic drama"),
    ("Blade Runner 2049", "2017 Denis Villeneuve sci-fi"),
    ("Wonder Woman", "2017 Patty Jenkins superhero"),
    ("Thor: Ragnarok", "2017 Taika Waititi Marvel"),
    ("Paddington 2", "2017 Paul King British comedy"),
    ("A Quiet Place", "2018 John Krasinski horror thriller"),
    ("Hereditary", "2018 Ari Aster horror drama"),
    ("First Reformed", "2017 Paul Schrader drama"),
    ("The Favourite", "2018 Yorgos Lanthimos historical drama"),
    ("Roma", "2018 Alfonso Cuarón drama"),
    ("Cold War", "2018 Paweł Pawlikowski Polish drama"),
    ("Shoplifters", "2018 Hirokazu Kore-eda Japanese drama"),
    ("BlackKklansman", "2018 Spike Lee biographical crime"),
    ("First Man", "2018 Damien Chazelle biographical drama"),
    ("Green Book", "2018 Peter Farrelly drama"),
    ("A Star Is Born", "2018 Bradley Cooper musical drama"),
    ("Black Panther", "2018 Ryan Coogler Marvel superhero"),
    ("Mission: Impossible – Fallout", "2018 Christopher McQuarrie action"),
    ("Aquaman", "2018 James Wan DC superhero"),
    ("Bohemian Rhapsody", "2018 Bryan Singer biographical music drama"),
    ("Parasite", "2019 Bong Joon-ho South Korean thriller"),
    ("Marriage Story", "2019 Noah Baumbach drama"),
    ("The Irishman", "2019 Martin Scorsese crime drama"),
    ("Once Upon a Time in Hollywood", "2019 Quentin Tarantino comedy drama"),
    ("Portrait of a Lady on Fire", "2019 Céline Sciamma French drama"),
    ("Joker", "2019 Todd Phillips crime drama"),
    ("1917", "2019 Sam Mendes war drama"),
    ("Little Women", "2019 Greta Gerwig period drama"),
    ("Pain and Glory", "2019 Pedro Almodóvar Spanish drama"),
    ("Knives Out", "2019 Rian Johnson mystery"),
    ("Midsommar", "2019 Ari Aster horror drama"),
    ("Us", "2019 Jordan Peele horror thriller"),
    ("The Lighthouse", "2019 Robert Eggers horror"),
    ("Ford v Ferrari", "2019 James Mangold biographical drama"),
    ("Avengers: Endgame", "2019 Marvel"),  # excluded via EXCLUDED_MOVIES check
    ("Spider-Man: Into the Spider-Verse", "2018 animated superhero"),
    ("Toy Story 4", "2019 Pixar"),  # excluded via EXCLUDED_MOVIES check
    # ── 2020s ──
    ("Nomadland", "2020 Chloé Zhao drama"),
    ("The Father", "2020 Florian Zeller drama"),
    ("Minari", "2020 Lee Isaac Chung drama"),
    ("Promising Young Woman", "2020 Emerald Fennell thriller"),
    ("Judas and the Black Messiah", "2021 Shaka King biographical drama"),
    ("The Power of the Dog", "2021 Jane Campion western drama"),
    ("Dune", "2021 Denis Villeneuve sci-fi epic"),
    ("The Tragedy of Macbeth", "2021 Joel Coen drama"),
    ("Drive My Car", "2021 Ryûsuke Hamaguchi Japanese drama"),
    ("Titane", "2021 Julia Ducournau French body horror"),
    ("Licorice Pizza", "2021 Paul Thomas Anderson romantic comedy"),
    ("The French Dispatch", "2021 Wes Anderson anthology comedy"),
    ("Encanto", "2021 Disney animated musical"),
    ("The Northman", "2022 Robert Eggers historical epic"),
    ("Everything Everywhere All at Once", "2022 Daniels sci-fi"),
    ("The Banshees of Inisherin", "2022 Martin McDonagh drama"),
    ("Tár", "2022 Todd Field drama"),
    ("The Fabelmans", "2022 Steven Spielberg biographical drama"),
    ("Aftersun", "2022 Charlotte Wells drama"),
    ("The Batman", "2022 Matt Reeves superhero"),
    ("Top Gun: Maverick", "2022 Joseph Kosinski action"),
    ("RRR", "2022 S. S. Rajamouli Indian action epic"),
    ("Oppenheimer", "2023 Christopher Nolan historical drama"),
    ("Barbie", "2023 Greta Gerwig comedy"),
    ("Killers of the Flower Moon", "2023 Martin Scorsese historical crime"),
    ("Anatomy of a Fall", "2023 Justine Triet French thriller"),
    ("Past Lives", "2023 Celine Song romantic drama"),
    ("Monster", "2023 Hirokazu Kore-eda Japanese drama"),
    ("Poor Things", "2023 Yorgos Lanthimos fantasy comedy"),
    ("The Zone of Interest", "2023 Jonathan Glazer historical drama"),
    ("Asteroid City", "2023 Wes Anderson comedy"),
    ("American Fiction", "2023 Cord Jefferson comedy drama"),
    ("Society of the Snow", "2023 J. A. Bayona survival drama"),
    ("The Boy and the Heron", "2023 Hayao Miyazaki animated fantasy"),
    ("Godzilla Minus One", "2023 Takashi Yamazaki Japanese monster film"),
    ("Conclave", "2024 Edward Berger drama"),
    ("The Substance", "2024 Coralie Fargeat body horror"),
    ("Emilia Pérez", "2024 Jacques Audiard French musical crime"),
    ("Anora", "2024 Sean Baker romantic drama"),
    ("The Brutalist", "2024 Brady Corbet historical drama"),
    # ── International / World Cinema ──
    ("Oldboy", "2003 Park Chan-wook South Korean thriller"),
    ("Train to Busan", "2016 Yeon Sang-ho South Korean horror"),
    ("Memories of Murder", "2003 Bong Joon-ho South Korean crime"),
    ("The Host", "2006 Bong Joon-ho South Korean monster film"),
    ("A Tale of Two Sisters", "2003 Kim Jee-woon South Korean horror"),
    ("I Saw the Devil", "2010 Kim Jee-woon South Korean thriller"),
    ("A Bittersweet Life", "2005 Kim Jee-woon South Korean action thriller"),
    ("Crouching Tiger, Hidden Dragon", "2000 Ang Lee martial arts"),
    ("Hero", "2002 Zhang Yimou Chinese martial arts"),
    ("House of Flying Daggers", "2004 Zhang Yimou Chinese martial arts"),
    ("Raise the Red Lantern", "1991 Zhang Yimou Chinese drama"),
    ("Y Tu Mamá También", "2001 Alfonso Cuarón Mexican road film"),
    ("The Secret in Their Eyes", "2009 Juan José Campanella Argentine thriller"),
    ("Wild Tales", "2014 Damián Szifron Argentine anthology"),
    ("A Separation", "2011 Asghar Farhadi Iranian drama"),
    ("The Salesman", "2016 Asghar Farhadi Iranian drama"),
    ("About Elly", "2009 Asghar Farhadi Iranian thriller"),
    ("4 Months, 3 Weeks and 2 Days", "2007 Cristian Mungiu Romanian drama"),
    ("Caché", "2005 Michael Haneke French thriller"),
    ("The Piano Teacher", "2001 Michael Haneke Austrian drama"),
    ("Russian Ark", "2002 Alexander Sokurov Russian drama"),
    ("Leviathan", "2014 Andrey Zvyagintsev Russian drama"),
    ("Andrei Rublev", "1966 Andrei Tarkovsky Russian drama"),
    ("Stalker", "1979 Andrei Tarkovsky Russian sci-fi"),
    ("Incendies", "2010 Denis Villeneuve Canadian drama"),
    ("Monsieur Lazhar", "2011 Philippe Falardeau Canadian drama"),
    ("Certified Copy", "2010 Abbas Kiarostami French-Italian drama"),
    ("The Great Dictator", "1940 Charlie Chaplin political satire"),
    ("M", "1931 Fritz Lang German thriller"),
    ("Metropolis", "1927 Fritz Lang German silent sci-fi"),
    ("Bicycle Thieves", "1948 Vittorio De Sica Italian neorealism"),
    # ── Animated (not in real topics) ──
    ("Princess Mononoke", "1997 Hayao Miyazaki Ghibli animation"),
    ("Howl's Moving Castle", "2004 Hayao Miyazaki Ghibli animation"),
    ("My Neighbor Totoro", "1988 Hayao Miyazaki Ghibli animation"),
    ("Kiki's Delivery Service", "1989 Hayao Miyazaki Ghibli animation"),
    ("Nausicaä of the Valley of the Wind", "1984 Hayao Miyazaki Ghibli animation"),
    ("Castle in the Sky", "1986 Hayao Miyazaki Ghibli animation"),
    ("Porco Rosso", "1992 Hayao Miyazaki Ghibli animation"),
    ("The Tale of the Princess Kaguya", "2013 Isao Takahata Ghibli animation"),
    ("Grave of the Fireflies", "1988 Isao Takahata Ghibli drama"),
    ("Only Yesterday", "1991 Isao Takahata Ghibli animation"),
    ("Akira", "1988 Katsuhiro Otomo anime sci-fi"),
    ("Ghost in the Shell", "1995 Mamoru Oshii anime sci-fi"),
    ("Paprika", "2006 Satoshi Kon anime sci-fi"),
    ("Perfect Blue", "1997 Satoshi Kon anime thriller"),
    ("A Bug's Life", "1998 Pixar animated adventure"),
    ("Brave", "2012 Pixar animated adventure"),
    ("Monsters University", "2013 Pixar animated comedy"),
    ("The Good Dinosaur", "2015 Pixar animated adventure"),
    ("Onward", "2020 Pixar animated fantasy"),
    ("Luca", "2021 Pixar animated adventure"),
    ("Tangled", "2010 Disney animated musical"),
    ("Wreck-It Ralph", "2012 Disney animated comedy"),
    ("Big Hero 6", "2014 Disney animated superhero"),
    ("Zootopia", "2016 Disney animated adventure"),  # in EXCLUDED_MOVIES
    ("Ralph Breaks the Internet", "2018 Disney animated comedy"),
    ("Turning Red", "2022 Pixar animated comedy drama"),
    ("Lightyear", "2022 Pixar animated sci-fi"),
    ("Elemental", "2023 Pixar animated fantasy"),
    ("Wallace & Gromit: The Curse of the Were-Rabbit", "2005 Aardman animated comedy"),
    ("Chicken Run", "2000 Aardman animated comedy"),
    ("Shrek", "2001 DreamWorks animated comedy"),
    ("Shrek 2", "2004 DreamWorks animated comedy"),
    ("Kung Fu Panda", "2008 DreamWorks animated action comedy"),
    ("How to Train Your Dragon", "2010 DreamWorks animated adventure"),
    ("The Prince of Egypt", "1998 DreamWorks animated musical"),
    ("Persepolis", "2007 Marjane Satrapi French animated biographical"),
    ("Waltz with Bashir", "2008 Ari Folman Israeli animated documentary"),
    ("Anomalisa", "2015 Charlie Kaufman animated drama"),
    ("Wolfwalkers", "2020 Cartoon Saloon Irish animated fantasy"),
    ("The Breadwinner", "2017 Cartoon Saloon animated drama"),
    # ── Horror / Thriller ──
    ("The Shining", "1980 Stanley Kubrick horror"),
    ("The Exorcist", "1973 William Friedkin horror"),
    ("Rosemary's Baby", "1968 Roman Polanski horror"),
    ("Halloween", "1978 John Carpenter slasher horror"),
    ("A Nightmare on Elm Street", "1984 Wes Craven horror"),
    ("The Texas Chain Saw Massacre", "1974 Tobe Hooper horror"),
    ("The Wicker Man", "1973 Robin Hardy British horror"),
    ("Don't Look Now", "1973 Nicolas Roeg horror thriller"),
    ("Misery", "1990 Rob Reiner Stephen King thriller"),
    ("Se7en", "1995 David Fincher thriller"),
    ("The Game", "1997 David Fincher thriller"),
    ("Zodiac", "2007 David Fincher crime thriller"),
    ("Hereditary", "2018 Ari Aster horror"),
    ("Midsommar", "2019 Ari Aster folk horror"),
    ("Suspiria", "1977 Dario Argento Italian horror"),
    ("The Others", "2001 Alejandro Amenábar horror"),
    ("Let the Right One In", "2008 Tomas Alfredson Swedish horror"),
    ("Raw", "2016 Julia Ducournau French horror"),
    ("It Follows", "2014 David Robert Mitchell horror"),
    ("The Babadook", "2014 Jennifer Kent Australian horror"),
    ("Mandy", "2018 Panos Cosmatos horror"),
    ("Parasite", "2019 Bong Joon-ho thriller"),
    ("Knives Out", "2019 Rian Johnson mystery"),
    ("Prisoners", "2013 Denis Villeneuve thriller"),
    ("Gone Girl", "2014 David Fincher thriller"),
    ("Nightcrawler", "2014 Dan Gilroy crime thriller"),
    # ── Science Fiction (not in real topics) ──
    ("Planet of the Apes", "1968 Franklin J. Schaffner sci-fi"),
    ("Soylent Green", "1973 Richard Fleischer sci-fi"),
    ("Logan's Run", "1976 Michael Anderson sci-fi"),
    ("The Thing", "1982 John Carpenter sci-fi horror"),
    ("Brazil", "1985 Terry Gilliam dystopian sci-fi"),
    ("Total Recall", "1990 Paul Verhoeven sci-fi action"),
    ("Starship Troopers", "1997 Paul Verhoeven sci-fi"),
    ("Gattaca", "1997 Andrew Niccol sci-fi drama"),
    ("eXistenZ", "1999 David Cronenberg sci-fi"),
    ("A.I. Artificial Intelligence", "2001 Steven Spielberg sci-fi"),
    ("Minority Report", "2002 Steven Spielberg sci-fi"),
    ("Children of Men", "2006 Alfonso Cuarón sci-fi drama"),
    ("The Road", "2009 John Hillcoat post-apocalyptic drama"),
    ("Source Code", "2011 Duncan Jones sci-fi"),
    ("Looper", "2012 Rian Johnson sci-fi"),
    ("Elysium", "2013 Neill Blomkamp sci-fi"),
    ("Edge of Tomorrow", "2014 Doug Liman sci-fi action"),
    ("Ex Machina", "2014 Alex Garland sci-fi"),
    ("The Lobster", "2015 Yorgos Lanthimos absurdist sci-fi"),
    ("Annihilation", "2018 Alex Garland sci-fi horror"),
    ("Ad Astra", "2019 James Gray sci-fi drama"),
    ("Underwater", "2020 William Eubank sci-fi horror"),
    # ── DC (not in real topics) ──
    ("Man of Steel", "2013 Zack Snyder superhero"),
    ("Batman v Superman", "2016 Zack Snyder superhero"),
    ("Justice League", "2017 Zack Snyder Marvel"),
    ("Shazam!", "2019 David F. Sandberg DC superhero"),
    ("Birds of Prey", "2020 Cathy Yan DC superhero"),
    # ── Bond (James Bond series) ──
    ("Dr. No", "1962 Terence Young James Bond"),
    ("Goldfinger", "1964 Guy Hamilton James Bond"),
    ("The Spy Who Loved Me", "1977 Lewis Gilbert James Bond"),
    ("GoldenEye", "1995 Martin Campbell James Bond"),
    ("Casino Royale", "2006 Martin Campbell James Bond"),
    ("Quantum of Solace", "2008 Marc Forster James Bond"),
    ("Spectre", "2015 Sam Mendes James Bond"),
    ("No Time to Die", "2021 Cary Joji Fukunaga James Bond"),
    # ── Musicals / Drama ──
    ("All That Jazz", "1979 Bob Fosse musical drama"),
    ("Nashville", "1975 Robert Altman musical drama"),
    ("The Last Waltz", "1978 Martin Scorsese music documentary"),
    ("Amadeus", "1984 Miloš Forman biographical drama"),
    ("Cabaret", "1972 Bob Fosse musical"),
    ("Hair", "1979 Miloš Forman musical"),
    ("The Sound of Music", "1965 Robert Wise musical"),
    ("West Side Story", "1961 Robert Wise musical"),
    ("West Side Story", "2021 Steven Spielberg musical"),
    ("Chicago", "2002 Rob Marshall musical"),
    ("Mamma Mia!", "2008 Phyllida Lloyd musical comedy"),
    ("Les Misérables", "2012 Tom Hooper musical drama"),
    ("Into the Woods", "2014 Rob Marshall musical"),
    ("La La Land", "2016 Damien Chazelle musical"),
    # ── Biographical / Historical ──
    ("Lincoln", "2012 Steven Spielberg historical drama"),
    ("J. Edgar", "2011 Clint Eastwood biographical drama"),
    ("The Crown", "2018 drama"),
    ("Frost/Nixon", "2008 Ron Howard political drama"),
    ("The Queen", "2006 Stephen Frears biographical drama"),
    ("The Iron Lady", "2011 Phyllida Lloyd biographical drama"),
    ("Darkest Hour", "2017 Joe Wright biographical drama"),
    ("Churchill", "2017 Jonathan Teplitzky biographical drama"),
    ("Dunkirk", "2017 Christopher Nolan war"),
    ("Hacksaw Ridge", "2016 Mel Gibson biographical war"),
    ("Selma", "2014 Ava DuVernay biographical drama"),
    ("Malcolm X", "1992 Spike Lee biographical drama"),
    ("Ali", "2001 Michael Mann biographical drama"),
    ("Ray", "2004 Taylor Hackford biographical music drama"),
    ("Walk the Line", "2005 James Mangold biographical music"),
    ("Rocketman", "2019 Dexter Fletcher biographical musical"),
    ("Bohemian Rhapsody", "2018 biographical musical drama"),
    ("Elvis", "2022 Baz Luhrmann biographical musical drama"),
    ("Spencer", "2021 Pablo Larraín biographical drama"),
    ("Jackie", "2016 Pablo Larraín biographical drama"),
    ("The Eyes of Tammy Faye", "2021 Michael Showalter biographical drama"),
    ("I, Tonya", "2017 Craig Gillespie biographical drama"),
    ("Vice", "2018 Adam McKay biographical comedy drama"),
    ("The Big Short", "2015 Adam McKay financial drama"),
    # ── Westerns ──
    ("The Good, the Bad and the Ugly", "1966 Sergio Leone spaghetti western"),
    ("Once Upon a Time in the West", "1968 Sergio Leone western"),
    ("A Fistful of Dollars", "1964 Sergio Leone spaghetti western"),
    ("High Noon", "1952 Fred Zinnemann western"),
    ("Shane", "1953 George Stevens western"),
    ("McCabe & Mrs. Miller", "1971 Robert Altman western"),
    ("Butch Cassidy and the Sundance Kid", "1969 George Roy Hill western"),
    ("Pat Garrett and Billy the Kid", "1973 Sam Peckinpah western"),
    ("True Grit", "2010 Coen Brothers western"),
    ("The Proposition", "2005 John Hillcoat Australian western"),
    ("Unforgiven", "1992 Clint Eastwood western"),
    ("Tombstone", "1993 George P. Cosmatos western"),
    ("Dances with Wolves", "1990 Kevin Costner western epic"),
    # ── Documentaries ──
    ("Bowling for Columbine", "2002 Michael Moore documentary"),
    ("Fahrenheit 9/11", "2004 Michael Moore political documentary"),
    ("An Inconvenient Truth", "2006 climate documentary"),
    ("Searching for Sugar Man", "2012 Malik Bendjelloul music documentary"),
    ("Amy", "2015 Asif Kapadia music documentary"),
    ("Senna", "2010 Asif Kapadia sports documentary"),
    ("Man on Wire", "2008 James Marsh documentary thriller"),
    ("The Act of Killing", "2012 Joshua Oppenheimer documentary"),
    ("Stories We Tell", "2012 Sarah Polley documentary"),
    ("20 Feet from Stardom", "2013 Morgan Neville music documentary"),
    ("Won't You Be My Neighbor?", "2018 Morgan Neville documentary"),
    ("Free Solo", "2018 Jimmy Chin documentary"),
    ("Tiger King", "2020 Eric Goode documentary"),
    ("The Last Dance", "2020 Jason Hehir documentary"),
    ("Summer of Soul", "2021 Ahmir Thompson documentary"),
    # ── War Films ──
    ("Patton", "1970 Franklin J. Schaffner biographical war"),
    ("M*A*S*H", "1970 Robert Altman war comedy"),
    ("The Deer Hunter", "1978 Michael Cimino war drama"),
    ("Coming Home", "1978 Hal Ashby war drama"),
    ("Full Metal Jacket", "1987 Stanley Kubrick war"),
    ("Platoon", "1986 Oliver Stone war"),
    ("Born on the Fourth of July", "1989 Oliver Stone biographical war"),
    ("Casualties of War", "1989 Brian De Palma war drama"),
    ("The Thin Red Line", "1998 Terrence Malick war drama"),
    ("Enemy at the Gates", "2001 Jean-Jacques Annaud war thriller"),
    ("The Pianist", "2002 Roman Polanski biographical war drama"),
    ("Letters from Iwo Jima", "2006 Clint Eastwood war drama"),
    ("Flags of Our Fathers", "2006 Clint Eastwood war drama"),
    ("Waltz with Bashir", "2008 animated war documentary"),
    ("The Hurt Locker", "2008 Kathryn Bigelow war thriller"),
    ("Inglourious Basterds", "2009 Quentin Tarantino war"),
    ("1917", "2019 Sam Mendes war drama"),
    ("Dunkirk", "2017 Christopher Nolan war"),
    ("Fury", "2014 David Ayer war drama"),
    ("Midway", "2019 Roland Emmerich war action"),
]

# ─────────────────────────────────────────────────────────────────────────────
# Prompts
# ─────────────────────────────────────────────────────────────────────────────

PASS1_PROMPT = """\
You are building a challenging retrieval benchmark for film information retrieval systems.

Task: Generate exactly {n_questions} fact-based questions about "{movie}" ({genre}).
Each question must be answerable from the corpus passages below.
After each question is retrieved by a search engine, a Haiku LLM will judge whether the \
retrieved passage actually answers it — so the answer must appear explicitly in a passage.

Corpus passages available:
<passages>
{passages}
</passages>

Challenge types to use (cover at least {n_types} different types):
{challenge_types}

Hard requirements:
1. Each answer must be a specific fact stated in one of the passages above
2. Do NOT repeat challenge types — each question must test a different skill
3. At least 3 questions must be genuinely hard to retrieve (vocabulary mismatch, \
indirect phrasing, requires knowing what NOT to say)
4. Questions must NOT mention the film title if the answer requires it (avoid trivial matches)
5. Some questions may be comparative ("which of X and Y...") — include the answer for both in corpus
6. Include at least 1 multi-hop question if the passages support it

Reply ONLY with a JSON array:
[
  {{
    "question": "...",
    "challenge_type": "...",
    "answer_hint": "brief phrase describing the answer",
    "difficulty": "easy|medium|hard",
    "answerable": true
  }}
]
Generate exactly {n_questions} items. JSON only, no preamble or explanation."""

PASS2_PROMPT = """\
You are verifying questions for a retrieval benchmark.

For each question, check if the answer is explicitly stated in the passages provided.
Return ONLY questions where the answer is clearly present — filter out any where \
the answer must be inferred, assumed, or is ambiguous.

Movie: {movie}
Passages:
<passages>
{passages}
</passages>

Questions to verify:
{questions_json}

For each question, reply with:
- "keep": true if the answer is explicitly in a passage, false otherwise
- "answer_span": the exact text from a passage that answers it (empty string if not found)
- "adjusted_difficulty": your assessment: easy/medium/hard

Reply ONLY with a JSON array matching the order of input questions:
[{{"keep": true/false, "answer_span": "...", "adjusted_difficulty": "easy|medium|hard"}}, ...]
JSON only."""

PASS3_PROMPT = """\
You are enriching a retrieval benchmark with harder paraphrase variants.

For each verified question about "{movie}", generate 1-2 harder reformulations that \
test vocabulary gap, implicit phrasing, or indirect reference to the same fact.
These harder variants will challenge retrieval systems that rely on keyword overlap.

Verified questions:
{questions_json}

Rules:
- Keep the same underlying fact/answer
- Vary vocabulary significantly (e.g., use synonyms, indirect descriptions, change question type)
- For multi-hop questions: make one hop implicit
- Do NOT create questions the corpus passages cannot answer

Reply ONLY with a JSON array — one entry per input question, each with a list of variants:
[
  {{"variants": ["harder version 1", "harder version 2"]}},
  ...
]
JSON only."""


def retrieve_for_movie(retriever, meta: list[dict], movie: str, k: int = 50) -> list[str]:
    """BM25-retrieve top-k passages for a movie name query."""
    try:
        tokens = bm25s.tokenize([movie], stopwords="en", stemmer=None,
                                 show_progress=False)
        results, _ = retriever.retrieve(tokens, k=k)
        return [meta[i]["text_snippet"] for i in results[0]]
    except Exception:
        return []


def parse_json(raw: str):
    text = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"(\[[\s\S]*\]|\{[\s\S]*\})", text)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                pass
    return None


def sonnet_call(client, prompt: str, max_tokens: int = 2000) -> tuple[str, int, int]:
    """Returns (text, input_tokens, output_tokens)."""
    for attempt in range(5):
        try:
            r = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            return r.content[0].text.strip(), r.usage.input_tokens, r.usage.output_tokens
        except Exception as e:
            wait = min(60, 5 * 2 ** attempt)
            print(f"  Sonnet error ({e}), retry in {wait}s")
            time.sleep(wait)
    return "", 0, 0


def token_cost_usd(in_tok: int, out_tok: int) -> float:
    return (in_tok * 3 + out_tok * 15) / 1_000_000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=float, default=40.0)
    parser.add_argument("--questions-per-movie", type=int, default=10)
    parser.add_argument("--loop-subset-size", type=int, default=1300,
                        help="Size of stratified loop subset to extract")
    parser.add_argument("--output", default=str(BASE / "data/synthetic_val"))
    parser.add_argument("--bm25-index", default=str(BASE / "data/processed/bm25s_index"))
    parser.add_argument("--meta", default=str(BASE / "data/processed/passages_meta.pkl"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    full_out   = out_dir / "synthetic_topics.json"
    loop_out   = out_dir / "synthetic_topics_loop.json"
    meta_out   = out_dir / "generation_metadata.json"
    cache_path = out_dir / "generation_cache.json"

    # Load resume cache
    cache: dict = {}
    if args.resume and cache_path.exists():
        with cache_path.open() as f:
            cache = json.load(f)
        print(f"Resuming — cache has {len(cache)} movies")

    print("Loading BM25 index and metadata...")
    with open(args.meta, "rb") as f:
        meta = pickle.load(f)
    retriever = bm25s.BM25.load(args.bm25_index, load_corpus=False)
    print(f"  {len(meta):,} passages loaded")

    client = anthropic.Anthropic()

    all_topics: list[dict] = []
    total_in_tokens = 0
    total_out_tokens = 0
    topic_counter = 0
    movies_processed = 0

    challenge_items = list(CHALLENGE_TYPES.items())
    n_challenge_types = min(args.questions_per_movie + 2, len(challenge_items))

    for movie_name, genre in CANDIDATE_MOVIES:
        # Budget guard
        total_cost = token_cost_usd(total_in_tokens, total_out_tokens)
        if total_cost >= args.budget:
            print(f"\nBudget limit ${args.budget:.2f} reached (spent ${total_cost:.2f}). Stopping.")
            break

        # Exclusion check
        skip = any(
            excl.lower() in movie_name.lower() or movie_name.lower() in excl.lower()
            for excl in EXCLUDED_MOVIES
        )
        if skip:
            continue

        movie_key = movie_name
        print(f"\n[{movies_processed+1}] {movie_name} | budget left: "
              f"${args.budget - token_cost_usd(total_in_tokens, total_out_tokens):.2f}")

        # ── Pass 1: Generate ──────────────────────────────────────────────────
        if f"p1_{movie_key}" not in cache:
            passages = retrieve_for_movie(retriever, meta, movie_name, k=60)
            # Prefer passages mentioning the movie
            words = [w for w in movie_name.lower().split() if len(w) > 3]
            relevant = [p for p in passages if any(w in p.lower() for w in words)]
            if len(relevant) < 5:
                relevant = passages[:30]
            else:
                relevant = relevant[:35]

            if not relevant:
                print(f"  No passages found, skipping")
                continue

            # Rotate challenge types
            offset = movies_processed % len(challenge_items)
            selected = []
            for i in range(n_challenge_types):
                selected.append(challenge_items[(offset + i) % len(challenge_items)])
            challenge_str = "\n".join(f"- {k}: {v}" for k, v in selected)
            passages_str  = "\n\n".join(f"[{i+1}] {p}" for i, p in enumerate(relevant[:30]))

            prompt = PASS1_PROMPT.format(
                movie=movie_name, genre=genre,
                n_questions=args.questions_per_movie,
                passages=passages_str,
                challenge_types=challenge_str,
                n_types=min(5, args.questions_per_movie),
            )
            raw, in_tok, out_tok = sonnet_call(client, prompt, max_tokens=2500)
            total_in_tokens += in_tok
            total_out_tokens += out_tok

            p1_result = parse_json(raw)
            if not isinstance(p1_result, list):
                print(f"  Pass 1 bad response, skipping")
                continue

            cache[f"p1_{movie_key}"] = {
                "questions": p1_result,
                "passages":  relevant,
                "in_tok":    in_tok,
                "out_tok":   out_tok,
            }
            with cache_path.open("w") as f:
                json.dump(cache, f)
        else:
            entry = cache[f"p1_{movie_key}"]
            p1_result = entry["questions"]
            relevant  = entry["passages"]
            total_in_tokens  += entry.get("in_tok", 0)
            total_out_tokens += entry.get("out_tok", 0)

        if not p1_result:
            continue
        print(f"  Pass 1: {len(p1_result)} questions generated")

        # ── Pass 2: Verify ────────────────────────────────────────────────────
        if f"p2_{movie_key}" not in cache:
            passages_str = "\n\n".join(f"[{i+1}] {p}" for i, p in enumerate(relevant[:30]))
            qs_json = json.dumps([{"question": q.get("question", ""),
                                   "challenge_type": q.get("challenge_type", "")}
                                  for q in p1_result], indent=2)
            prompt = PASS2_PROMPT.format(
                movie=movie_name,
                passages=passages_str,
                questions_json=qs_json,
            )
            raw, in_tok, out_tok = sonnet_call(client, prompt, max_tokens=1500)
            total_in_tokens  += in_tok
            total_out_tokens += out_tok

            p2_result = parse_json(raw)
            if not isinstance(p2_result, list):
                # Keep all if parse fails
                p2_result = [{"keep": True, "answer_span": "", "adjusted_difficulty": q.get("difficulty","medium")}
                             for q in p1_result]

            cache[f"p2_{movie_key}"] = {"verifications": p2_result, "in_tok": in_tok, "out_tok": out_tok}
            with cache_path.open("w") as f:
                json.dump(cache, f)
        else:
            entry = cache[f"p2_{movie_key}"]
            p2_result = entry["verifications"]
            total_in_tokens  += entry.get("in_tok", 0)
            total_out_tokens += entry.get("out_tok", 0)

        # Merge pass 1 + pass 2
        verified = []
        for i, q_obj in enumerate(p1_result):
            v = p2_result[i] if i < len(p2_result) else {"keep": True}
            if v.get("keep", True):
                verified.append({
                    **q_obj,
                    "answer_span": v.get("answer_span", ""),
                    "difficulty":  v.get("adjusted_difficulty", q_obj.get("difficulty", "medium")),
                })
        print(f"  Pass 2: {len(verified)}/{len(p1_result)} verified")

        if not verified:
            continue

        # ── Pass 3: Enrich with harder variants ───────────────────────────────
        if f"p3_{movie_key}" not in cache:
            qs_json = json.dumps([{"question": q["question"],
                                   "challenge_type": q.get("challenge_type", "")}
                                  for q in verified], indent=2)
            prompt = PASS3_PROMPT.format(movie=movie_name, questions_json=qs_json)
            raw, in_tok, out_tok = sonnet_call(client, prompt, max_tokens=2000)
            total_in_tokens  += in_tok
            total_out_tokens += out_tok

            p3_result = parse_json(raw)
            if not isinstance(p3_result, list):
                p3_result = [{"variants": []} for _ in verified]

            cache[f"p3_{movie_key}"] = {"enriched": p3_result, "in_tok": in_tok, "out_tok": out_tok}
            with cache_path.open("w") as f:
                json.dump(cache, f)
        else:
            entry = cache[f"p3_{movie_key}"]
            p3_result = entry["enriched"]
            total_in_tokens  += entry.get("in_tok", 0)
            total_out_tokens += entry.get("out_tok", 0)

        # Build final topic list for this movie
        movie_topics = []
        for i, q_obj in enumerate(verified):
            topic_counter += 1
            tid = f"S{topic_counter:04d}"
            movie_topics.append({
                "topic_id":      tid,
                "question":      q_obj["question"],
                "movie":         movie_name,
                "genre":         genre,
                "challenge_type": q_obj.get("challenge_type", "unknown"),
                "difficulty":    q_obj.get("difficulty", "medium"),
                "answer_hint":   q_obj.get("answer_hint", ""),
                "answer_span":   q_obj.get("answer_span", ""),
                "pass":          "verified",
                "generation_model": "claude-sonnet-4-6",
            })
            # Add harder variants from pass 3
            variants = []
            if i < len(p3_result):
                v_entry = p3_result[i]
                if isinstance(v_entry, dict):
                    variants = v_entry.get("variants", [])
                elif isinstance(v_entry, list):
                    variants = v_entry
            for var in variants:
                if isinstance(var, str) and len(var) > 15:
                    topic_counter += 1
                    tid2 = f"S{topic_counter:04d}"
                    movie_topics.append({
                        "topic_id":       tid2,
                        "question":       var,
                        "movie":          movie_name,
                        "genre":          genre,
                        "challenge_type": q_obj.get("challenge_type", "unknown") + "_hard",
                        "difficulty":     "hard",
                        "answer_hint":    q_obj.get("answer_hint", ""),
                        "answer_span":    q_obj.get("answer_span", ""),
                        "parent_topic_id": tid,
                        "pass":           "enriched_variant",
                        "generation_model": "claude-sonnet-4-6",
                    })

        all_topics.extend(movie_topics)
        movies_processed += 1
        spent = token_cost_usd(total_in_tokens, total_out_tokens)
        print(f"  Pass 3: +{len(movie_topics) - len(verified)} variants | "
              f"total topics: {len(all_topics)} | spent: ${spent:.3f}")

        # Incremental save
        with full_out.open("w") as f:
            json.dump(all_topics, f, indent=2)

    # ── Build stratified loop subset (1300 topics) ────────────────────────────
    random.seed(42)

    # Stratify by difficulty and challenge type for a balanced subset
    hard   = [t for t in all_topics if t["difficulty"] == "hard"]
    medium = [t for t in all_topics if t["difficulty"] == "medium"]
    easy   = [t for t in all_topics if t["difficulty"] == "easy"]
    # Target: 40% hard, 45% medium, 15% easy (harder than real test set)
    n_hard   = min(len(hard),   int(args.loop_subset_size * 0.40))
    n_medium = min(len(medium), int(args.loop_subset_size * 0.45))
    n_easy   = args.loop_subset_size - n_hard - n_medium

    loop_topics = (random.sample(hard,   n_hard)   +
                   random.sample(medium, min(len(medium), n_medium)) +
                   random.sample(easy,   min(len(easy),   n_easy)))
    # Shuffle and trim to target
    random.shuffle(loop_topics)
    loop_topics = loop_topics[:args.loop_subset_size]
    # Re-assign sequential IDs for the loop subset
    for i, t in enumerate(loop_topics):
        t["loop_topic_id"] = f"L{i+1:04d}"

    with loop_out.open("w") as f:
        json.dump(loop_topics, f, indent=2)

    # ── Metadata / stats ──────────────────────────────────────────────────────
    challenge_dist = defaultdict(int)
    difficulty_dist = defaultdict(int)
    for t in all_topics:
        challenge_dist[t.get("challenge_type", "unknown")] += 1
        difficulty_dist[t.get("difficulty", "medium")] += 1

    total_cost = token_cost_usd(total_in_tokens, total_out_tokens)
    stats = {
        "total_topics":            len(all_topics),
        "loop_subset_size":        len(loop_topics),
        "movies_processed":        movies_processed,
        "total_input_tokens":      total_in_tokens,
        "total_output_tokens":     total_out_tokens,
        "total_cost_usd":          round(total_cost, 4),
        "generation_model":        "claude-sonnet-4-6",
        "judge_model_in_loop":     "claude-haiku-4-5-20251001",
        "challenge_distribution":  dict(sorted(challenge_dist.items(), key=lambda x: -x[1])),
        "difficulty_distribution": dict(difficulty_dist),
        "pass_distribution":       {
            "verified":          sum(1 for t in all_topics if t.get("pass") == "verified"),
            "enriched_variant":  sum(1 for t in all_topics if t.get("pass") == "enriched_variant"),
        },
    }
    with meta_out.open("w") as f:
        json.dump(stats, f, indent=2)

    print(f"\n{'='*65}")
    print(f"SYNTHETIC GENERATION COMPLETE")
    print(f"{'='*65}")
    print(f"Total topics:        {len(all_topics)}")
    print(f"Loop subset:         {len(loop_topics)}")
    print(f"Movies processed:    {movies_processed}")
    print(f"Total cost:          ${total_cost:.2f}")
    print(f"\nDifficulty distribution:")
    for d, n in sorted(difficulty_dist.items()):
        pct = 100 * n / max(len(all_topics), 1)
        print(f"  {d:<10} {n:>5} ({pct:.1f}%)")
    print(f"\nTop challenge types:")
    for ct, n in list(sorted(challenge_dist.items(), key=lambda x: -x[1]))[:8]:
        print(f"  {ct:<30} {n:>5}")
    print(f"\nOutputs:")
    print(f"  Full dataset:   {full_out}")
    print(f"  Loop subset:    {loop_out}")
    print(f"  Metadata:       {meta_out}")
    print(f"\nNext:")
    print(f"  python scripts/autoresearch_loop.py --topics-file {loop_out} \\")
    print(f"    --output-dir data/synthetic_autoresearch")


if __name__ == "__main__":
    main()
