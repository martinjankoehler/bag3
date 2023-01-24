;; SPDX-License-Identifier: BSD-3-Clause AND Apache-2.0
;; Copyright 2018 Regents of the University of California
;; All rights reserved.
;;
;; Redistribution and use in source and binary forms, with or without
;; modification, are permitted provided that the following conditions are met:
;;
;; * Redistributions of source code must retain the above copyright notice, this
;;   list of conditions and the following disclaimer.
;;
;; * Redistributions in binary form must reproduce the above copyright notice,
;;   this list of conditions and the following disclaimer in the documentation
;;   and/or other materials provided with the distribution.
;;
;; * Neither the name of the copyright holder nor the names of its
;;   contributors may be used to endorse or promote products derived from
;;   this software without specific prior written permission.
;;
;; THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
;; AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
;; IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
;; DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
;; FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
;; DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
;; SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
;; CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
;; OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
;; OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

;; Copyright 2019 Blue Cheetah Analog Design Inc.
;;
;; Licensed under the Apache License, Version 2.0 (the "License");
;; you may not use this file except in compliance with the License.
;; You may obtain a copy of the License at
;;
;;     http://www.apache.org/licenses/LICENSE-2.0
;;
;; Unless required by applicable law or agreed to in writing, software
;; distributed under the License is distributed on an "AS IS" BASIS,
;; WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
;; See the License for the specific language governing permissions and
;; limitations under the License.

/*  Note:

Due to licensing reasons, this skill script is missing the function
CCSinvokeCdfCallbacks() from Cadence solution 11018344, which executes
CDF parameters callback from skill.

If you do not need to instantiate a pcell instance, this method
is not needed.

Eric Chang, Mar 2, 2017.

*/


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;  Virtuoso Database operations functions  ;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; reads a skill data structure from file
procedure( parse_data_from_file( fname "t" )
    let( (p ans)
        unless( p = infile( fname )
            error("Cannot open file %s" fname)
        )
        ans = parse_data_from_file_helper(p)
        close( p )
        ans
    )
)

; recursive helper for parse_data_from_file
procedure( parse_data_from_file_helper( p )
    let( (line item ans finish key)
        gets( line p )
        ; remove newline
        line = substring(line 1 strlen(line) - 1)
        ; printf("read line: %s\n" line)
        cond(
            (line == "#list"
            ; parse a list
            ans = tconc(nil 0)
            while( nequal(item = parse_data_from_file_helper(p) "#end")
                tconc(ans item)
            )
            ; printf("returning list ")
            ; print(cdar(ans))
            ; printf("\n")
            cdar(ans)
            )
            (line == "#prop_list"
            ; parse a disembodied property list
            ans = ncons(nil)
            finish = nil
            while( !finish
                key = parse_data_from_file_helper(p)
                if( key == "#end" then
                    finish = 't
                else
                    item = parse_data_from_file_helper(p)
                    putprop(ans item key)
                )
            )
            ans
            )
            ; parse a float
            (strncmp( line "#float" 6 ) == 0
                cdfParseFloatString(cadr(parseString(line)))
            )
            ; parse an int
            (strncmp( line "#int" 4 ) == 0
                atoi(cadr(parseString(line)))
            )
            ; parse a boolean
            (strncmp( line "#bool" 5 ) == 0
                if( atoi(cadr(parseString(line))) == 1 then
                    't
                else
                    nil
                )
            )
            ; parse a string token or #end
            ('t
                ; printf("returning str %s\n" line)
                line
            )
        )
    )
)

; This function is used to print the bounding box in parse_cad_sch()
procedure( print_bbox(p indent bbox dbu_per_uu)
    fprintf( p "%sbbox:\n" indent )
    foreach( pair bbox
        foreach( elem pair
            fprintf( p "%s  - %d\n" indent round(elem * dbu_per_uu) )
        )
    )
)

; This function is used to print the signal enum in parse_cad_sch()
procedure( print_sig_type(p indent sig_type)
    let( (sig_enum)
        if(sig_type == "signal" then
            sig_enum = 0
        else if(sig_type == "supply" then
            sig_enum = 1
        else    ; sig_type == "ground"
            sig_enum = 2
        ))
        fprintf( p "%sstype: %d\n" indent sig_enum )
    )
)

; This function is used to print the term enum using direction in parse_cad_sch()
procedure( print_term_type_from_direction(p indent direction)
    let( (term_enum)
        if(direction == "inputOutput" then
            term_enum = 2
        else if(direction == "input" then
            term_enum = 0
        else    ; direction == "output"
            term_enum = 1
        ))
        fprintf( p "%sttype: %d\n" indent term_enum )
    )
)

; This function is used to print the term enum using name in parse_cad_sch()
procedure( print_term_type_from_name(p indent name)
    let( (term_enum)
        if(name == "iopin" then
            term_enum = 2
        else if(name == "ipin" then
            term_enum = 0
        else    ; name == "opin"
            term_enum = 1
        ))
        fprintf( p "%sttype: %d\n" indent term_enum )
    )
)

; This function is used to print the transformation in parse_cad_sch()
procedure( print_xform(p indent xform dbu_per_uu)
    let( (xform_idx)
        fprintf( p "%sxform:\n" indent )
        xform_idx = 0
        foreach( xf xform
            if(xform_idx == 1 then
                fprintf( p "%s  - %s\n" indent xf)
            else if(xform_idx == 0 then
                foreach( xy xf
                    fprintf( p "%s  - %d\n" indent round(xy * dbu_per_uu))
                )
            ))
            xform_idx++
        )
    )
)

; This function is used to print "is_primitive" in parse_cad_sch()
procedure( print_is_primitive(p indent lib_name)
    let( (is_primitive)
        if( lib_name == "basic" || lib_name == "analogLib" || lib_name == "ahdlLib" || lib_name == "BAG_prim" then
            is_primitive = "true"
        else
            is_primitive = "false"
        )
        fprintf( p "%sis_primitive: %s\n" indent is_primitive )
    )
)

; This function is used to print the origin in parse_cad_sch()
procedure( print_origin(p indent origin dbu_per_uu)
    fprintf( p "%sorigin:\n" indent )
    foreach( xy origin
        fprintf( p "%s  - %d\n" indent round(xy * dbu_per_uu) )
    )
)

; This function is used to print the alignment enum in parse_cad_sch()
procedure( print_alignment(p indent align_type)
    let( (align_enum)
        if( align_type == "centerRight" then
            align_enum = 7
        else if( align_type == "centerLeft" then
            align_enum = 1
        else if( align_type == "centerCenter" then
            align_enum = 4
        else if( align_type == "lowerLeft" then
            align_enum = 2
        else if( align_type == "lowerRight" then
            align_enum = 8
        else if( align_type == "lowerCenter" then
            align_enum = 5
        else if( align_type == "upperLeft" then
            align_enum = 0
        else if( align_type == "upperRight" then
            align_enum = 6
        else if( align_type == "upperCenter" then
            align_enum = 3
        else
            error( "Unrecognized justify = %s" align_type )
        )))))))))
        fprintf( p "%salignment: %d\n" indent align_enum )
    )
)

; This function is used to print the font enum in parse_cad_sch()
procedure( print_font(p indent font_type)
    let( (font_enum)
        if( font_type == "stick" then
            font_enum = 5
        else
            error( "Unrecognized font = %s" font_type )
        )
        fprintf( p "%sfont: %d\n" indent font_enum )
    )
)

; This function is used to print boolean values in parse_cad_sch()
procedure( print_bool(p indent attr attr_name)
    if( attr then
        fprintf( p "%s%s: true\n" indent attr_name )
    else
        fprintf( p "%s%s: false\n" indent attr_name )
    )
)

defMathConstants('MyConsts)
; Parse the netlist of the given cellview.
; Works on schematic and veriloga and symbol.
procedure( parse_cad_sch(lib_name cell_name view_name file_name "ttt")
    let( (skip cv p indent shp_status shp_ptr inst term_cnt inst_cnt prop_cnt obj_type path_enum)
        skip = 0
        if(view_name == "schematic" then
            unless( cv = dbOpenCellViewByType( lib_name cell_name view_name nil "r" )
                view_name = "veriloga"
                unless( cv = dbOpenCellViewByType( lib_name cell_name view_name nil "r" )
                    error( "Cannot find schematic or veriloga view of cell %s__%s" lib_name cell_name )
                )
            )
        else
            unless( cv = dbOpenCellViewByType( lib_name cell_name view_name nil "r" )
                skip = 1
            )
        )
        if(skip == 0 then
            indent = ""
            p = outfile( file_name "w" )

            ; print cellview information
            printf( "*INFO* Writing cell %s__%s (%s) netlist to %s\n" lib_name cell_name view_name file_name )
            fprintf( p "%slib_name: %s\n" indent lib_name )
            fprintf( p "%scell_name: %s\n" indent cell_name )
            fprintf( p "%sview_name: %s\n" indent view_name )

            print_bbox(p indent cv~>bBox cv~>DBUPerUU)

            ; print terminals
            indent = ""
            term_cnt = 0
            fprintf( p "%sterminals:\n" indent )
            if(view_name == "symbol" then
                ; use cv~>shapes instead of cv~>terminals since the first has more information
                foreach( shape cv~>shapes
                    ; pick the shape only if it is (pin, drawing)
                    if(shape~>layerNum == 229 && shape~>purpose == "drawing" then
                        term_cnt++

                        ; print entry for terminal
                        indent = "  "
                        fprintf( p "%s%s:\n" indent shape~>net~>name )
                        ; print terminal information
                        indent = "    "
                        fprintf( p "%sobj:\n" indent )
                        indent = "      "
                        fprintf( p "%s- 0\n" indent )   ; enum for symbol terminals
                        fprintf( p "%s- layer: %d\n" indent shape~>layerNum )
                        fprintf( p "%s  purpose: 4294967295\n" indent )     ; get purpose number from cv?
                        fprintf( p "%s  net: \"\"\n" indent )

                        indent = "        "
                        print_bbox(p indent shape~>bBox cv~>DBUPerUU)

                        indent = "    "
                        print_sig_type(p indent shape~>net~>sigType)
                        print_term_type_from_direction(p indent shape~>pin~>term~>direction)
                    )
                )
            else
                ; use cv~>shapes instead of cv~>terminals since the first has more information
                foreach( shape cv~>shapes
                    if(shape~>layerNum == 229 then
                        term_cnt++

                        ; print entry for terminal
                        indent = "  "
                        fprintf( p "%s%s:\n" indent shape~>associate~>name )
                        ; print terminal information
                        indent = "    "
                        fprintf( p "%sobj:\n" indent )
                        indent = "      "
                        fprintf( p "%s- 1\n" indent )   ; enum for schematic terminals
                        fprintf( p "%s- inst:\n" indent )

                        ; find corresponding instance in cv~>instances, and exit once match is found
                        shp_status = t
                        shp_ptr = cv~>instances
                        while(shp_status && shp_ptr
                            inst = car(shp_ptr)
                            if(inst~>net~>name == shape~>associate~>name then
                                ; found match, exit
                                shp_status = nil
                            else
                                shp_ptr = cdr(shp_ptr)
                            )
                        )

                        indent = "          "
                        fprintf( p "%slib_name: %s\n" indent inst~>libName )
                        fprintf( p "%scell_name: %s\n" indent inst~>cellName )
                        fprintf( p "%sview_name: %s\n" indent inst~>viewName )

                        print_xform(p indent inst~>transform cv~>DBUPerUU)
                        print_bbox(p indent inst~>bBox cv~>DBUPerUU)

                        fprintf( p "%sconnections:\n" indent )
                        fprintf( p "%s  {}\n" indent )
                        fprintf( p "%sparams:\n" indent )
                        fprintf( p "%s  {}\n" indent )

                        print_is_primitive(p indent inst~>libName)

                        ; print attr
                        indent = "      "
                        fprintf( p "%s  attr:\n" indent )
                        indent = "          "
                        fprintf( p "%slayer: %d\n" indent shape~>layerNum )
                        fprintf( p "%spurpose: 237\n" indent )  ; get purpose number from cv?
                        fprintf( p "%snet: \"\"\n" indent )
                        print_origin(p indent shape~>xy cv~>DBUPerUU)

                        print_alignment(p indent shape~>justify)

                        fprintf( p "%sorient: %s\n" indent shape~>orient )
                        print_font(p indent shape~>font)
                        fprintf( p "%sheight: %d\n" indent round(shape~>height * cv~>DBUPerUU) )

                        print_bool(p indent shape~>isOverbar "overbar")
                        print_bool(p indent shape~>isVisible "visible")
                        print_bool(p indent shape~>isDrafted "drafting")
                        if( shape~>assocType == "attrAssocType" then
                            fprintf( p "%sattr_type: 0\n" indent )
                        else
                            error( "Unrecognized assocType = %s" shape~>assocType )
                        )
                        fprintf( p "%sformat: 1\n" indent )     ; get from cv?

                        indent = "    "
                        print_sig_type(p indent inst~>net~>sigType)
                        print_term_type_from_name(p indent inst~>cellName)
                    )
                )
            )
            when(term_cnt == 0
                fprintf( p "  {}\n" )
            )

            indent = ""
            fprintf( p "%sshapes:\n" indent )
            foreach( shape cv~>shapes
                ; if schematic, pick only if it's not pin
                ; if symbol, pick only if it's not (pin, drawing)
                if( (view_name == "schematic" && shape~>layerNum != 229) || (view_name == "symbol" && !(shape~>layerNum == 229 && shape~>purpose == "drawing")) then
                    indent = "  "
                    fprintf( p "%s-\n" indent )

                    indent = "    "
                    if( shape~>purpose == "label" then
                        if( shape~>layerName == "wire" || shape~>layerName == "pin" then
                            obj_type = 7   ; enum for wire label and pin label
                        else if( shape~>layerName == "device" || shape~>layerName == "instance" then
                            obj_type = 8   ; enum for device label and instance label
                        else
                            error( "Unrecognized layerName = %s" shape~>layerName )
                        ))
                        fprintf( p "%s- %d\n" indent obj_type )
                        fprintf( p "%s- layer: %d\n" indent shape~>layerNum )
                        fprintf( p "%s  purpose: 237\n" indent )    ; get purpose number from cv?

                        indent = "      "
                        if( shape~>net~>name then
                            fprintf( p "%snet: %s\n" indent shape~>net~>name )
                        else
                            fprintf( p "%snet: \"\"\n" indent )
                        )
                        print_origin(p indent shape~>xy cv~>DBUPerUU)
                        print_alignment(p indent shape~>justify)

                        fprintf( p "%sorient: %s\n" indent shape~>orient )
                        print_font(p indent shape~>font)
                        fprintf( p "%sheight: %d\n" indent round(shape~>height * cv~>DBUPerUU) )
                        print_bool(p indent shape~>isOverbar "overbar")
                        print_bool(p indent shape~>isVisible "visible")
                        print_bool(p indent shape~>isDrafted "drafting")

                        if( shape~>layerName == "device" || shape~>layerName == "instance" then
                            fprintf( p "%stext: \"%s\"\n" indent shape~>theLabel )
                            fprintf( p "%sevaluator: cdsNLPEvalText\n" indent )   ; get from cv?
                        else
                            if(shape~>theLabel then
                                fprintf( p "%stext: %s\n" indent shape~>theLabel )
                            else
                                fprintf( p "%stext: %s\n" indent shape~>net~>name )
                            )
                        )
                    else
                        if( shape~>objType == "line" then
                            obj_type = 5   ; enum for wire drawing and some device drawing
                        else if( shape~>objType == "path" then
                            obj_type = 6   ; enum for some device drawing1
                        else if( shape~>objType == "arc" then
                            obj_type = 2   ; enum for some device drawing
                        else if( shape~>objType == "rect" then
                            obj_type = 0   ; enum for instance drawing and some device drawing
                        else if( shape~>objType == "ellipse" then
                            obj_type = 4   ; enum for some device drawing1 and some device drawing
                        else
                            error( "Unrecognized objType = %s" shape~>objType )
                        )))))
                        fprintf( p "%s- %d\n" indent obj_type )
                        fprintf( p "%s- layer: %d\n" indent shape~>layerNum )
                        if( shape~>purpose == "drawing" then
                            fprintf( p "%s  purpose: 4294967295\n" indent )     ; get purpose number from cv?
                        else if( shape~>purpose == "drawing1" then
                            fprintf( p "%s  purpose: 241\n" indent )    ; get purpose number from cv?
                        else
                            error( "Unexpected purpose = %s" shape~>purpose )
                        ))
                        if( shape~>net then
                            fprintf( p "%s  net: %s\n" indent shape~>net~>name )
                        else
                            fprintf( p "%s  net: \"\"\n" indent )
                        )

                        if( shape~>objType == "line" || shape~>objType == "path" then
                            if( shape~>objType == "path" then
                                fprintf( p "%s  width: %d\n" indent round(shape~>width * cv~>DBUPerUU) )
                            )
                            fprintf( p "%s  points:\n" indent )
                            foreach( pair shape~>points
                                fprintf( p "%s    -\n" indent )
                                foreach( elem pair
                                    fprintf( p "%s      - %d\n" indent round(elem * cv~>DBUPerUU) )
                                )
                            )
                            if( shape~>objType == "path" then
                                if( shape~>pathStyle == "roundRound" then
                                    path_enum = 2    ; enum for round path style
                                else
                                    error( "Unrecognized path style = %s" shape~>pathStyle )
                                )
                                fprintf( p "%s  style: %d\n" indent path_enum )
                                fprintf( p "%s  begin_ext: %d\n" indent round(shape~>beginExt * cv~>DBUPerUU) )
                                fprintf( p "%s  end_ext: %d\n" indent round(shape~>endExt * cv~>DBUPerUU) )
                            )
                        else
                            if( shape~>objType == "arc" then
                                if( shape~>startAngle < 0 then
                                    fprintf( p "%s  ang_start: %.16f\n" indent 2 * MyConsts.PI + shape~>startAngle )
                                else
                                    fprintf( p "%s  ang_start: %.16f\n" indent shape~>startAngle )
                                )
                                if( shape~>stopAngle < 0 then
                                    fprintf( p "%s  ang_stop: %.16f\n" indent 2 * MyConsts.PI + shape~>stopAngle )
                                else
                                    fprintf( p "%s  ang_stop: %.16f\n" indent shape~>stopAngle )
                                )
                            )
                            indent = "      "
                            print_bbox(p indent shape~>bBox cv~>DBUPerUU)
                        )
                    )
                )
            )

            ; print instances
            indent = ""
            inst_cnt = 0
            fprintf( p "%sinstances:\n" indent )
            foreach( inst cv~>instances
                if(inst~>purpose != "pin" then
                    inst_cnt++

                    ; print entry for instance
                    indent = "  "
                    fprintf( p "%s%s:\n" indent inst~>name )
                    ; print instance master information.
                    indent = "    "
                    fprintf( p "%slib_name: %s\n" indent inst~>libName )
                    fprintf( p "%scell_name: %s\n" indent inst~>cellName )
                    fprintf( p "%sview_name: %s\n" indent inst~>viewName )

                    print_xform(p indent inst~>transform cv~>DBUPerUU)
                    print_bbox(p indent inst~>bBox cv~>DBUPerUU)

                    ; print instance terminal information
                    if( !(inst~>instTerms) then
                        fprintf( p "%sconnections: {}\n" indent )
                    else
                        fprintf( p "%sconnections:\n" indent )
                        foreach( inst_term inst~>instTerms
                            indent = "      "
                            fprintf( p "%s%s: %s\n" indent inst_term~>name inst_term~>net~>name )
                        )
                    )
                    indent = "    "

                    ; print params
                    fprintf( p "%sparams:\n" indent )
                    prop_cnt = 0
                    foreach(param inst~>prop
                        fprintf( p "%s  %s:\n" indent param~>name )
                        fprintf( p "%s    - 3\n" indent ) ; enum for string
                        if(param~>value == "" then
                            fprintf( p "%s    - \"\"\n" indent )
                        else
                            fprintf( p "%s    - %s\n" indent param~>value )
                        )
                        prop_cnt++
                    )
                    when(prop_cnt == 0
                        fprintf( p "%s  {}\n" indent )
                    )

                    print_is_primitive(p indent inst~>libName)
                )
            )
            when(inst_cnt == 0
                fprintf( p "  {}\n" )
            )

            ; print properties
            indent = ""
            fprintf( p "%sprops:\n" indent )
            foreach( _prop cv~>prop
                indent = "  "
                fprintf( p "%s%s:\n" indent _prop~>name )
                indent = "    "
                if( _prop~>valueType == "string" then
                    fprintf( p "%s- 3\n" indent )   ; enum for string
                    if( _prop~>value == "" then
                        fprintf( p "%s- \"\"\n" indent )
                    else
                        fprintf( p "%s- %s\n" indent _prop~>value )
                    )
                else if( _prop~>valueType == "int" then
                    fprintf( p "%s- 0\n" indent )   ; enum for int
                    fprintf( p "%s- %d\n" indent _prop~>value )
                else if( _prop~>valueType == "time" then
                    fprintf( p "%s- 4\n" indent )   ; enum for time
                    fprintf( p "%s- time_val: %d\n" indent stringToTime(_prop~>value) )
                else if( _prop~>valueType == "ILList" then
                    fprintf( p "%s- 5\n" indent )   ; enum for ILList
                    fprintf( p "%s- name: ILList\n" indent )
                    fprintf( p "%s  bin_val: (" indent )
                    prop_cnt = 0
                    foreach( _elem _prop~>value
                        if( prop_cnt > 0
                            fprintf( p " " )
                        )
                        fprintf( p "\"%s\"" _elem )
                        prop_cnt++
                    )
                    fprintf( p ")\n" )
                ))))
            )

            ; print app_defs
            indent = ""
            fprintf( p "%sapp_defs:\n" indent )
            fprintf( p "%s  _dbLastSavedCounter:\n" indent )
            fprintf( p "%s    - 0\n" indent )
            fprintf( p "%s    - %d\n" indent cv~>modifiedCounter )
            fprintf( p "%s  _dbvCvTimeStamp:\n" indent )
            fprintf( p "%s    - 0\n" indent )
            fprintf( p "%s    - %d\n" indent cv~>modifiedCounter )
            fprintf( p "%s  cdbRevision:\n" indent )
            fprintf( p "%s    - 0\n" indent )
            fprintf( p "%s    - 227612\n" indent )  ; get from cv?
            ; TODO: get cdnSPDesignMajorVersion

            ; close resources
            close(p)
            dbClose(cv)
        )
    )
)


; close all opened cellviews
procedure( close_all_cellviews()
    foreach( cv dbGetOpenCellViews()
        dbPurge(cv)
    )
    't
)


; release write locks from all the given cellviews
procedure( release_write_locks( lib_name cell_view_list_f "tt" )
    let( (cell_view_list lib_obj cv)
        cell_view_list = parse_data_from_file(cell_view_list_f)
        when( lib_obj = ddGetObj(lib_name nil nil nil nil "r")
            foreach( info cell_view_list
                when( cv = dbFindOpenCellView( lib_obj car(info) cadr(info) )
                    dbReopen(cv, "r")
                    dbClose(cv)
                )
            )
            ddReleaseObj(lib_obj)
        )
        t
    )
)


; refresh all given cell views
procedure( refresh_cellviews( lib_name cell_view_list_f "tt" )
    let( (cell_view_list lib_obj cv)
        cell_view_list = parse_data_from_file(cell_view_list_f)
        when( lib_obj = ddGetObj(lib_name nil nil nil nil "r")
            foreach( info cell_view_list
                when( cv = dbFindOpenCellView( lib_obj car(info) cadr(info) )
                    dbRefreshCellView(cv)
                )
            )
            ddReleaseObj(lib_obj)
        )
        t
    )
)


; if library with lib_name does not exist, create a new library with that name.
; Otherwise, if erase is true, remove all cells in that library.
; Returns the library database object.
procedure( create_or_erase_library(lib_name tech_lib lib_path erase "tttg")
    let( (lib_obj)
        if( lib_obj = ddGetObj(lib_name nil nil nil nil "r") then
            when( erase
                ; delete all cells in the library
                foreach( cell lib_obj~>cells
                    unless( ddDeleteObj(cell)
                        error("cannot delete cell %s in library %s\n" cell~>name lib_name)
                    )
                )
            )
            ddReleaseObj(lib_obj)
            't
        else
            ; create library if not exist
            when( and(lib_path (lib_path != "."))
                lib_path = strcat(lib_path "/" lib_name)
            )
            lib_obj = ddCreateLib(lib_name lib_path)
            ; attach technology file
            techBindTechFile(lib_obj tech_lib)
            ; close library
            ddReleaseObj(lib_obj)
            't
        )
    )
)


; modify label text in a layout
procedure( modify_label(lib_name cell_name reg_before reg_after "tttt")
    let((cv lbl_txt new_lbl)
        unless( cv = dbOpenCellViewByType( lib_name cell_name "layout" nil "a" )
            error( "Cannot find layout view of cell %s__%s" lib_name cell_name )
        )

        foreach(shape_obj cv~>shapes
            when( lbl_txt = shape_obj->theLabel
                when( rexMatchp(reg_before lbl_txt)
                    new_lbl = rexReplace(lbl_txt reg_after 1)
                    shape_obj->theLabel = new_lbl
                )
            )
        )

        dbSave(cv)
        dbClose(cv)
    )
)


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;  BAG server related functions            ;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

procedure( stdoutHandler(ipcId data)
    let( (result result_str)
        if( bag_server_started > 0 then
            printf("*INFO* Evaluate expression from BAG process: %s\n" data)
            if( result = errsetstring(data 't) then
                sprintf(result_str "%A\n" car(result))
            else
                sprintf(result_str "%s\n" car(nthelem(5 errset.errset)))
            )
            printf("*INFO* Sending result to BAG process: %s" result_str)
            ipcWriteProcess(ipcId sprintf(nil "%d\n" strlen(result_str)))
            ipcWriteProcess(ipcId result_str)
            't
        else
            if( data == "BAG skill server has started.  Yay!\n" then
                bag_server_started = 1
                printf("*INFO* BAG skill server started.\n")
            else
                printf("*INFO* Waiting for BAG skill server.  Message: %s\n" data)
            )
        )
    )
)

procedure( stderrHandler(ipcId data)
    warn("BAG server process error: %s\n" data)
    warn("Shutting down BAG server.")
    ipcKillProcess(ipcId)
    't
)

procedure( exitHandler(ipcId exitId)
    printf("*INFO* BAG server process exited with status: %d\n" exitId)
    't
)

procedure( start_bag_server()
    bag_server_started = 0
    printf("*INFO* Starting BAG server process.\n")
    ipcBeginProcess("bash virt_server.sh" "" 'stdoutHandler 'stderrHandler 'exitHandler "")
)

bag_server_started = 0
bag_modify_adexl_counter = 0
bag_proc = start_bag_server()
